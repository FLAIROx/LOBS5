#!/usr/bin/env python3
"""
Diagnostic script to identify import-time hang in E2 validation.
Prints at each import step to locate the bottleneck.
"""

import sys
import os
import time

# Timing helper
start = time.time()
def log(msg):
    elapsed = time.time() - start
    print(f"[{elapsed:7.2f}s] {msg}", flush=True)

log("Script started")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
log("Added project root to path")

# Step 1: Basic imports
log("Importing sys, os, time - done (already imported)")

# Step 2: JAX imports
log("Importing jax...")
import jax
log(f"jax imported - version {jax.__version__}")

log("Calling jax.devices()...")
devices = jax.devices()
log(f"jax.devices() returned: {devices}")

log("Importing jax.numpy...")
import jax.numpy as jnp
log("jax.numpy imported")

# Step 3: Dataclasses
log("Importing dataclasses...")
from dataclasses import dataclass
log("dataclass imported")

# Step 4: Test simple JAX operation
log("Testing jnp.ones((10,10))...")
x = jnp.ones((10, 10))
log(f"Created array with shape {x.shape}")

log("Testing x @ x.T...")
y = x @ x.T
y.block_until_ready()
log(f"Matrix multiply completed, shape {y.shape}")

# Step 5: es_lobs5 package
log("Importing es_lobs5 package...")
import es_lobs5
log("es_lobs5 imported")

# Step 6: es_lobs5.training
log("Importing es_lobs5.training...")
from es_lobs5 import training
log("es_lobs5.training imported")

# Step 7: ESTrainer (this is the suspicious one)
log("Importing es_lobs5.training.es_trainer...")
from es_lobs5.training.es_trainer import ESTrainer
log("ESTrainer imported!")

# Step 8: Quick test
log("Creating minimal ESConfig class...")

@dataclass
class MinimalConfig:
    lobs5_checkpoint: str = "/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/"
    noiser: str = 'eggroll'
    sigma: float = 0.01
    lr: float = 0.001
    lora_rank: int = 4
    grad_clip: float = 1.0
    n_threads: int = 256
    n_epochs: int = 1
    n_steps: int = 10
    world_msgs_per_step: int = 5
    token_mode: int = 24
    background_mode: str = 'historical_replay'
    replay_data_path: str = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
    data_dir: str = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2021"
    task: str = 'sell'
    task_size: int = 500
    tick_size: int = 100
    seed: int = 42
    output_dir: str = '/tmp/es_validation_e2'

log("MinimalConfig class created")

config = MinimalConfig()
log("Config instance created")

# Step 9: Initialize trainer (this likely triggers XLA compilation)
log("Initializing ESTrainer (may take time for XLA compilation)...")
try:
    trainer = ESTrainer(config)
    log("ESTrainer initialized successfully!")
except Exception as e:
    log(f"ESTrainer initialization failed: {e}")
    import traceback
    traceback.print_exc()

log("=== Import diagnostic complete ===")
