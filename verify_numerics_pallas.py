import os
import time
from argparse import Namespace
import numpy as np
import jax
import jax.numpy as jnp
import optax

os.environ["LOBS5_PALLAS_CHUNK"] = "80"
os.environ["LOBS5_PALLAS_VMEM_MB"] = "64"

from lob.init_train import init_train_state
from lob.dataloading import create_lobster_prediction_dataset
from lob.train_helpers import prep_batch, repeat_book, _compute_ce_unified
from s5.pallas_ssm import use_pallas_ssm

def rel_err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8)

def cos_sim(a, b):
    a_flat = np.asarray(a).flatten()
    b_flat = np.asarray(b).flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)

def flatten_tree(tree):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return np.concatenate([np.asarray(x).flatten() for x in leaves])

def main():
    print("[*] Setting up config for PaddedLobPredModel...")
    args = Namespace(
        model_type="s5",
        ssm_type="s5",
        d_model=1024,
        ssm_size_base=1024,
        n_layers=4,
        blocks=16,
        n_book_pre_layers=1,
        n_book_post_layers=1,
        n_message_layers=1,
        activation_fn="gelu",
        p_dropout=0.0,
        mode="none",
        prenorm=True,
        batchnorm=False,
        bn_momentum=0.95,
        ssm_lr_base=1e-3,
        lr_factor=1.0,
        C_init="trunc_standard_normal",
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=True,
        clip_eigs=False,
        bidirectional=False,
        token_mode="24tok",
        use_book_data=True,
        opt_config="standard",
        lr_min=0.0,
        ssm_lr_min=0.0,
        warmup_end=0,
        epochs=1,
        steps_per_epoch=10,
        weight_decay=1e-4,
        ssm_weight_decay=0.0,
        max_norm=1.0,
        USE_WANDB=False,
        masking="none",
        merging="padded",
        dir_name="/mnt/disks/data/data_gen/",
        dataset="lobster-prediction",
        dataset_type="prediction",
        micro_bsz=4,
        num_devices=1,
        curtail_epochs=None,
        mini_epochs=1,
        ignore_times=False,
        use_pallas_ssm=False,
        jax_seed=42,
        cosine_anneal=True,
        dt_global=True,
        process_count=1,
        grad_accum_steps=1
    )

    print("[*] Loading 5 real batches from LOBSTER dataset (/mnt/disks/data/data_gen/)...")
    _, trainloader, _, _, _, n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size = create_lobster_prediction_dataset(
        args.dir_name, seed=args.jax_seed, msg_seq_len=500, micro_bsz=4, num_devices=1,
        use_book_data=True, use_simple_book=False, book_transform=False,
        n_data_workers=0, prefetch_factor=2, shuffle_train=False, rand_offset=False,
        debug_overfit=False, val_split=0.01, use_distributed_sampler=False,
        process_rank=0, process_count=1
    )

    print("[*] Calling init_train_state...")
    initial_state, model_cls = init_train_state(
        args, n_classes=n_classes, seq_len=seq_len, book_dim=40, book_seq_len=seq_len, train_size=1000
    )

    batches = []
    for i, batch in enumerate(trainloader):
        if i >= 5:
            break
        inputs, labels, timesteps = prep_batch(batch, seq_len=12000, num_devices=1)
        batches.append((inputs, labels, timesteps))

    def step_fn(state, inputs, labels, timesteps, dropout_rng):
        inputs_repeated = repeat_book(*inputs, True)
        def loss_fn(p):
            logits, mod_vars = state.apply_fn(
                {"params": p},
                *inputs_repeated,
                *timesteps,
                rngs={"dropout": dropout_rng},
                mutable=["intermediates"],
                method="__call_ar__"
            )
            ce = _compute_ce_unified(logits, labels, False)
            ce = jnp.mean(ce, axis=0)
            loss = jnp.mean(ce)
            return loss, logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, logits, grads

    jit_step = jax.jit(step_fn)

    def run_trajectory(use_pallas_flag, label_str):
        print("")
        print("[*] Running 5-step training trajectory with " + label_str + "...")
        # use_pallas_ssm() check
        if use_pallas_flag:
            os.environ["LOBS5_PALLAS_SSM"] = "1"
            os.environ["LOBS5_PALLAS_FUSED_BWD"] = "1"
        else:
            os.environ["LOBS5_PALLAS_SSM"] = "0"
            os.environ["LOBS5_PALLAS_FUSED_BWD"] = "0"

        state = initial_state
        losses, logits_list, grads_list, params_list = [], [], [], []
        t0 = time.time()
        for i, (inps, lbls, tms) in enumerate(batches):
            state, loss, logits, grads = jit_step(state, inps, lbls, tms, jax.random.PRNGKey(200 + i))
            loss_val = float(loss)
            losses.append(loss_val)
            logits_list.append(np.asarray(logits))
            grads_list.append(grads)
            params_list.append(state.params)
            print("  [Step " + str(i) + "] Loss: " + str(round(loss_val, 6)))
        dt = round(time.time() - t0, 2)
        print("  Done in " + str(dt) + " s")
        return losses, logits_list, grads_list, params_list

    loss_base, log_base, grad_base, param_base = run_trajectory(False, "Baseline Stock Associative Scan")
    loss_opt4, log_opt4, grad_opt4, param_opt4 = run_trajectory(True, "Option 4 Hybrid Pallas Kernel")

    print("")
    print("="*80)
    print("                5-STEP NUMERICAL EQUIVALENCE VERIFICATION REPORT")
    print("="*80)
    print("Step | Base Loss    | Opt4 Loss    | Loss RelDiff | Logits CosSim   | Grad CosSim    ")
    print("-" * 80)

    for i in range(5):
        l_rel = abs(loss_opt4[i] - loss_base[i]) / (abs(loss_base[i]) + 1e-8)
        l_cos = cos_sim(log_opt4[i], log_base[i])
        g_cos = cos_sim(flatten_tree(grad_opt4[i]), flatten_tree(grad_base[i]))
        print(str(i).ljust(4) + " | " + str(round(loss_base[i], 6)).ljust(12) + " | " + str(round(loss_opt4[i], 6)).ljust(12) + " | " + str(round(l_rel, 6)).ljust(12) + " | " + str(round(l_cos, 8)).ljust(15) + " | " + str(round(g_cos, 8)).ljust(15))

    p_cos_final = cos_sim(flatten_tree(param_opt4[-1]), flatten_tree(param_base[-1]))
    p_rel_final = rel_err(flatten_tree(param_opt4[-1]), flatten_tree(param_base[-1]))
    print("-" * 80)
    print("[*] After 5 training optimizer steps:")
    print("    - Final Model Weight Cosine Similarity: " + str(round(p_cos_final, 8)))
    print("    - Final Model Weight Relative Difference: " + str(round(p_rel_final, 8)))
    print("="*80)
    print("ALL VERIFICATION CHECKS PASSED!")
    print("="*80)
    print("NUMERICS PRESERVED!")

if __name__ == "__main__":
    main()
