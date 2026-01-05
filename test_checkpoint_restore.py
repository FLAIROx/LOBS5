#!/usr/bin/env python3
"""
Test script for LOBS5 mid-epoch checkpoint save/restore functionality.

This script tests:
1. Step-level checkpoint saving with Orbax
2. Optimizer state (including momentum) persistence
3. Proper step counter restoration
4. Training state integrity after restore

Usage:
    # Run in lobs5 conda environment
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5
    JAX_PLATFORMS=cpu python test_checkpoint_restore.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Force CPU mode for fast testing
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_BF16"] = "0"  # Disable BF16 for CPU testing

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import orbax.checkpoint as ocp

# Add LOBS5 to path
LOBS5_ROOT = Path(__file__).parent
sys.path.insert(0, str(LOBS5_ROOT))


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def inspect_checkpoint_structure(ckpt_dir: str) -> dict:
    """Inspect checkpoint directory structure and contents."""
    ckpt_path = Path(ckpt_dir)
    result = {
        "path": str(ckpt_path),
        "exists": ckpt_path.exists(),
        "contents": [],
        "steps": [],
        "metadata": None,
        "state_structure": None,
    }

    if not ckpt_path.exists():
        return result

    # List all items in checkpoint directory
    for item in sorted(ckpt_path.iterdir()):
        result["contents"].append(str(item.name))

        # Check if this is a step directory (numeric)
        if item.is_dir() and item.name.isdigit():
            result["steps"].append(int(item.name))

            # Look for metadata
            metadata_path = item / "metadata" / "_ROOT_METADATA"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        result["metadata"] = json.load(f)
                except Exception as e:
                    result["metadata"] = f"Error reading: {e}"

            # Look for state structure
            state_path = item / "state"
            if state_path.exists():
                state_contents = []
                for state_item in state_path.rglob("*"):
                    if state_item.is_file():
                        rel_path = state_item.relative_to(state_path)
                        state_contents.append(str(rel_path))
                result["state_structure"] = state_contents[:30]  # First 30 files

    result["steps"] = sorted(result["steps"])
    return result


def get_all_optimizer_stats(opt_state) -> dict:
    """Recursively extract all optimizer state statistics."""
    stats = {}

    def extract_stats(obj, prefix=""):
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                extract_stats(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                extract_stats(value, f"{prefix}.{key}" if prefix else key)
        elif hasattr(obj, 'shape'):
            # This is a JAX array
            stats[prefix] = {
                "shape": list(obj.shape),
                "mean": float(jnp.mean(obj)),
                "std": float(jnp.std(obj)),
                "min": float(jnp.min(obj)),
                "max": float(jnp.max(obj)),
            }

    try:
        # For multi_transform, iterate inner_states
        if hasattr(opt_state, 'inner_states'):
            for group_name, group_state in opt_state.inner_states.items():
                extract_stats(group_state, f"inner_states.{group_name}")
        else:
            extract_stats(opt_state, "opt_state")
    except Exception as e:
        stats["error"] = str(e)

    return stats


def inspect_optimizer_state(state) -> dict:
    """Extract optimizer state information for verification."""
    opt_info = {
        "step": int(state.step),
        "optimizer_groups": [],
        "has_mu_nu": False,
    }

    try:
        # Get all leaves of opt_state
        leaves = jax.tree_util.tree_leaves(state.opt_state)
        leaf_info = []
        has_nonzero = False

        for i, leaf in enumerate(leaves[:20]):  # First 20 leaves
            if hasattr(leaf, 'shape'):
                mean_val = float(jnp.mean(leaf))
                if abs(mean_val) > 1e-10:
                    has_nonzero = True
                leaf_info.append({
                    "index": i,
                    "shape": list(leaf.shape),
                    "mean": mean_val,
                })

        opt_info["leaf_samples"] = leaf_info
        opt_info["total_leaves"] = len(leaves)
        opt_info["has_nonzero_values"] = has_nonzero

        # Handle optax.MultiTransform structure
        if hasattr(state.opt_state, 'inner_states'):
            inner_states = state.opt_state.inner_states
            for group_name, group_state in inner_states.items():
                group_info = {
                    "name": group_name,
                    "type": type(group_state).__name__,
                    "inner_type": None,
                }

                if hasattr(group_state, 'inner_state'):
                    inner = group_state.inner_state
                    group_info["inner_type"] = type(inner).__name__

                    # Check for ScaleByAdamState-like structure (mu, nu, count)
                    if hasattr(inner, 'count'):
                        group_info["count"] = int(inner.count)

                    # Try to find mu/nu in various places
                    for attr_name in ['mu', 'nu', 'trace', 'second_moment']:
                        if hasattr(inner, attr_name):
                            attr = getattr(inner, attr_name)
                            if attr is not None:
                                leaves = jax.tree_util.tree_leaves(attr)
                                if leaves:
                                    first_leaf = leaves[0]
                                    if hasattr(first_leaf, 'shape'):
                                        group_info[f"has_{attr_name}"] = True
                                        group_info[f"{attr_name}_mean"] = float(jnp.mean(first_leaf))
                                        opt_info["has_mu_nu"] = True

                opt_info["optimizer_groups"].append(group_info)

    except Exception as e:
        opt_info["error"] = str(e)
        import traceback
        opt_info["traceback"] = traceback.format_exc()

    return opt_info


def compare_optimizer_states(state1, state2) -> dict:
    """Compare two optimizer states to verify restoration."""
    comparison = {
        "step_match": int(state1.step) == int(state2.step),
        "step1": int(state1.step),
        "step2": int(state2.step),
        "leaves_match": True,
        "params_match": True,
        "differences": [],
    }

    try:
        # Compare all leaves of opt_state
        leaves1 = jax.tree_util.tree_leaves(state1.opt_state)
        leaves2 = jax.tree_util.tree_leaves(state2.opt_state)

        comparison["opt_leaves_count1"] = len(leaves1)
        comparison["opt_leaves_count2"] = len(leaves2)

        if len(leaves1) != len(leaves2):
            comparison["leaves_match"] = False
            comparison["differences"].append(f"Leaf count mismatch: {len(leaves1)} vs {len(leaves2)}")
        else:
            for i, (l1, l2) in enumerate(zip(leaves1, leaves2)):
                if hasattr(l1, 'shape') and hasattr(l2, 'shape'):
                    if l1.shape != l2.shape:
                        comparison["leaves_match"] = False
                        comparison["differences"].append(f"opt_leaf[{i}] shape: {l1.shape} vs {l2.shape}")
                    elif not jnp.allclose(l1, l2, rtol=1e-5, atol=1e-8):
                        comparison["leaves_match"] = False
                        max_diff = float(jnp.max(jnp.abs(l1 - l2)))
                        comparison["differences"].append(f"opt_leaf[{i}] max_diff={max_diff}")

        # Compare params
        params1_leaves = jax.tree_util.tree_leaves(state1.params)
        params2_leaves = jax.tree_util.tree_leaves(state2.params)

        for i, (p1, p2) in enumerate(zip(params1_leaves, params2_leaves)):
            if not jnp.allclose(p1, p2, rtol=1e-5, atol=1e-8):
                comparison["params_match"] = False
                comparison["differences"].append(
                    f"params[{i}]: max_diff={float(jnp.max(jnp.abs(p1 - p2)))}"
                )
                if len(comparison["differences"]) > 10:
                    comparison["differences"].append("... (truncated)")
                    break

    except Exception as e:
        comparison["error"] = str(e)
        import traceback
        comparison["traceback"] = traceback.format_exc()

    return comparison


def generate_report(results: dict, output_path: str):
    """Generate markdown report from test results."""
    report = f"""# LOBS5 Checkpoint Save/Restore Test Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Summary

| Test | Status |
|------|--------|
| Checkpoint Directory Created | {"PASS" if results.get("checkpoint_created") else "FAIL"} |
| Step-level Saves | {"PASS" if results.get("step_saves_found") else "FAIL"} |
| Optimizer State Saved | {"PASS" if results.get("opt_state_saved") else "FAIL"} |
| State Restored Correctly | {"PASS" if results.get("state_restored") else "FAIL"} |
| Optimizer State Match | {"PASS" if results.get("opt_state_match") else "FAIL"} |
| Step Counter Match | {"PASS" if results.get("step_match") else "FAIL"} |
| Params Match | {"PASS" if results.get("params_match") else "FAIL"} |

## Test Configuration

```
Platform: {results.get("platform", "N/A")}
Target Steps: {results.get("target_steps", "N/A")}
Checkpoint Interval: Every {results.get("checkpoint_interval", "N/A")} steps
Model: Simple test model (mimics LOBS5 optimizer structure)
```

## Checkpoint Structure

### Directory Contents
```
{json.dumps(results.get("checkpoint_structure", {}), indent=2)}
```

### Saved Steps
Steps found: {results.get("saved_steps", "N/A")}

## Optimizer State Analysis

### Before Save (at step {results.get("save_step", "N/A")})
```json
{json.dumps(results.get("opt_state_before", {}), indent=2)}
```

### After Restore
```json
{json.dumps(results.get("opt_state_after", {}), indent=2)}
```

## State Comparison

```json
{json.dumps(results.get("state_comparison", {}), indent=2)}
```

## Key Findings

### Optimizer State Persistence
The test verifies that the following are correctly saved and restored:
- **state.step**: Training step counter (critical for LR schedule)
- **opt_state leaves**: All optimizer state arrays (including Adam mu/nu moments)
- **params**: Model parameters

### What This Means for Training Resumption
When training is interrupted mid-epoch and restored:
1. The step counter is restored, so the learning rate schedule continues correctly
2. The optimizer state is restored, so gradient history is preserved
3. Training can continue seamlessly without "cold start" issues

## Conclusions

"""

    all_pass = all([
        results.get("checkpoint_created", False),
        results.get("step_saves_found", False),
        results.get("opt_state_saved", False),
        results.get("state_restored", False),
        results.get("opt_state_match", False),
        results.get("step_match", False),
        results.get("params_match", False),
    ])

    if all_pass:
        report += "**All tests PASSED.** The mid-epoch checkpoint save/restore functionality is working correctly.\n\n"
        report += "The optimizer state is properly preserved, allowing training to resume without loss of optimization history.\n"
    else:
        report += "**Some tests FAILED.** See details above.\n\n"

        if not results.get("checkpoint_created"):
            report += "- Checkpoint directory was not created properly.\n"
        if not results.get("opt_state_saved"):
            report += "- Optimizer state was not saved in the checkpoint.\n"
        if not results.get("opt_state_match"):
            report += "- Optimizer state was not restored correctly.\n"
        if not results.get("step_match"):
            report += "- Step counter was not restored correctly.\n"
        if not results.get("params_match"):
            report += "- Model parameters were not restored correctly.\n"

    # Write report
    with open(output_path, "w") as f:
        f.write(report)

    log(f"Report written to: {output_path}")
    return report


def test_checkpoint_with_lobs5_structure():
    """
    Test checkpoint save/restore with LOBS5's actual optimizer structure.
    Uses optax.multi_transform like LOBS5 does.
    """
    from lob.train_helpers import create_lobs5_learning_rate_schedule

    results = {
        "platform": str(jax.devices()),
        "target_steps": 50,
        "checkpoint_interval": 20,
        "logs": "",
    }

    log("Starting LOBS5-style checkpoint test")
    log(f"JAX devices: {jax.devices()}")

    # Create temporary checkpoint directory
    test_ckpt_dir = tempfile.mkdtemp(prefix="lobs5_ckpt_test_")
    log(f"Using temp checkpoint dir: {test_ckpt_dir}")

    try:
        # Create LOBS5-style optimizer setup (mirrors train_helpers.py)
        log("Creating LOBS5-style optimizer...")

        # Create learning rate schedules (like LOBS5)
        ssm_lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=5e-5,
            warmup_end_step=50,
            total_steps=500,
            lr_min=0.0,
            use_cosine_anneal=True,
        )
        lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=5e-5,
            warmup_end_step=50,
            total_steps=500,
            lr_min=0.0,
            use_cosine_anneal=True,
        )

        # Create mock params structure (mimics LOBS5 model)
        mock_params = {
            "encoder": {
                "embedding": jnp.zeros((44, 256)),
                "layers_0": {
                    "B": jnp.zeros((64, 128)),
                    "C": jnp.zeros((128, 64)),
                    "Lambda_re": jnp.zeros((64,)),
                    "Lambda_im": jnp.zeros((64,)),
                    "log_step": jnp.zeros((1,)),
                    "norm": jnp.zeros((256,)),
                    "dense": jnp.zeros((256, 256)),
                }
            },
            "decoder": {
                "kernel": jnp.zeros((256, 44)),
                "bias": jnp.zeros((44,)),
            }
        }

        # Create param label function (like LOBS5's ssm_fn)
        def map_nested_fn(fn):
            def map_fn(nested_dict):
                return {
                    k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
                    for k, v in nested_dict.items()
                }
            return map_fn

        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
            else ("none" if k in [] else "regular")
        )

        # Create multi_transform optimizer (like LOBS5)
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": optax.adam(learning_rate=ssm_lr_schedule),
                "regular": optax.adamw(learning_rate=lr_schedule, weight_decay=0.05),
            },
            ssm_fn,
        )

        # Create TrainState
        log("Creating TrainState...")
        state = train_state.TrainState.create(
            apply_fn=lambda params, x: x,  # dummy
            params=mock_params,
            tx=tx,
        )

        log(f"Initial step: {state.step}")

        # Inspect initial optimizer state
        opt_info_initial = inspect_optimizer_state(state)
        log(f"Initial optimizer state: {json.dumps(opt_info_initial, indent=2)}")

        # Simulate training steps
        log("Simulating 50 training steps...")
        dummy_grads = jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x) * 0.01,
            state.params
        )

        for step in range(50):
            state = state.apply_gradients(grads=dummy_grads)
            if (step + 1) % 10 == 0:
                log(f"Step {step + 1}, state.step = {state.step}")

        log(f"After training: step = {state.step}")

        # Inspect optimizer state after training
        opt_info_after_train = inspect_optimizer_state(state)
        log(f"Optimizer state after training: {json.dumps(opt_info_after_train, indent=2)}")
        results["opt_state_before"] = opt_info_after_train
        results["save_step"] = int(state.step)

        # Now test with LOBS5's actual save/load functions
        from lob.init_train import save_checkpoint as lobs5_save_checkpoint

        log("Creating checkpoint manager...")
        mgr_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            create=True,
            max_to_keep=5,
        )
        ckpt_mgr = ocp.CheckpointManager(
            test_ckpt_dir,
            item_names=('state', 'metadata'),
            options=mgr_options,
        )

        # Save checkpoint using LOBS5's function
        log("Saving checkpoint with LOBS5 save_checkpoint()...")
        ckpt = {
            'model': state,  # Note: no deduplicate_trainstate on CPU
            'config': {"test": True, "step": int(state.step)},
            'metrics': {
                'loss_train': 0.5,
                'epoch': 0,
                'step': int(state.step),
            }
        }

        save_result = lobs5_save_checkpoint(ckpt_mgr, ckpt, int(state.step))
        log(f"Save result: {save_result}")

        # Wait for async save
        ckpt_mgr.wait_until_finished()

        # Inspect checkpoint structure
        log("Inspecting checkpoint structure...")
        ckpt_structure = inspect_checkpoint_structure(test_ckpt_dir)
        log(f"Checkpoint structure: {json.dumps(ckpt_structure, indent=2)}")
        results["checkpoint_structure"] = ckpt_structure
        results["checkpoint_created"] = ckpt_structure["exists"]
        results["saved_steps"] = ckpt_structure["steps"]
        results["step_saves_found"] = len(ckpt_structure["steps"]) > 0

        # Check if opt_state is in the checkpoint by looking for characteristic files
        if ckpt_structure["state_structure"]:
            # Orbax saves arrays in ocdbt format, look for any array data
            has_array_data = any("ocdbt" in s or "checkpoint" in s or ".npy" in s
                                for s in ckpt_structure["state_structure"])
            results["opt_state_saved"] = has_array_data
            log(f"Array data in checkpoint: {has_array_data}")
        else:
            results["opt_state_saved"] = False

        # Create fresh state for restore test
        log("Creating fresh state for restore test...")
        fresh_state = train_state.TrainState.create(
            apply_fn=lambda params, x: x,
            params=mock_params,
            tx=tx,
        )
        log(f"Fresh state step: {fresh_state.step}")

        # Load checkpoint using Orbax directly with abstract_pytree approach
        log("Loading checkpoint...")

        # Create a new checkpoint manager for loading
        load_mgr = ocp.CheckpointManager(
            os.path.abspath(test_ckpt_dir),
            item_names=('state', 'metadata'),
            options=ocp.CheckpointManagerOptions(),
        )

        # Get the step to restore
        latest_step = load_mgr.latest_step()
        log(f"Latest checkpoint step: {latest_step}")

        # Restore with abstract pytree matching saved structure
        restored = load_mgr.restore(
            latest_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state),  # Use the saved state structure
                metadata=ocp.args.JsonRestore()
            )
        )

        restored_state = restored['state']
        restored_metadata = restored['metadata']

        log(f"Restored state step: {restored_state.step}")
        log(f"Restored metadata: {restored_metadata}")

        # Inspect restored optimizer state
        opt_info_restored = inspect_optimizer_state(restored_state)
        log(f"Restored optimizer state: {json.dumps(opt_info_restored, indent=2)}")
        results["opt_state_after"] = opt_info_restored

        # Compare states
        log("Comparing states...")
        comparison = compare_optimizer_states(state, restored_state)
        log(f"State comparison: {json.dumps(comparison, indent=2)}")
        results["state_comparison"] = comparison

        results["state_restored"] = True
        results["step_match"] = comparison["step_match"]
        results["opt_state_match"] = comparison["leaves_match"]
        results["params_match"] = comparison["params_match"]

        log("Test completed successfully!")

    except Exception as e:
        import traceback
        error_msg = f"Error: {e}\n{traceback.format_exc()}"
        log(error_msg, "ERROR")
        results["logs"] = error_msg
        results["checkpoint_created"] = results.get("checkpoint_created", False)
        results["state_restored"] = False
        results["opt_state_match"] = False
        results["params_match"] = False
        results["step_match"] = False

    finally:
        # Cleanup
        log(f"Cleaning up temp directory: {test_ckpt_dir}")
        shutil.rmtree(test_ckpt_dir, ignore_errors=True)

    return results


def main():
    log("=" * 60)
    log("LOBS5 Checkpoint Save/Restore Test")
    log("=" * 60)

    # Run test with LOBS5 optimizer structure
    results = test_checkpoint_with_lobs5_structure()

    # Generate report
    report_path = str(LOBS5_ROOT / "checkpoint_test_report.md")
    generate_report(results, report_path)

    # Print summary
    log("=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)

    all_pass = all([
        results.get("checkpoint_created", False),
        results.get("step_saves_found", False),
        results.get("opt_state_saved", False),
        results.get("state_restored", False),
        results.get("opt_state_match", False),
        results.get("step_match", False),
        results.get("params_match", False),
    ])

    if all_pass:
        log("ALL TESTS PASSED!", "SUCCESS")
        return 0
    else:
        log("SOME TESTS FAILED!", "FAILURE")
        for key in ["checkpoint_created", "step_saves_found", "opt_state_saved",
                    "state_restored", "opt_state_match", "step_match", "params_match"]:
            status = "PASS" if results.get(key, False) else "FAIL"
            log(f"  {key}: {status}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
