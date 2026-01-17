
import wandb
import argparse

# =============================================================================
# ES Sweep Configuration
# =============================================================================
# Fixed parameters based on user constraints
# - task_size: 50
# - background_msgs_per_step: 50
# - n_steps: 50
# - pergpu_perturbations: 32 (Default)
# =============================================================================

sweep_config = {
    'program': 'es_lobs5/scripts/es_training.py',
    'method': 'grid',  # Grid search for small parameter space
    'metric': {
        'name': 'train/return_mean',  # Maximizing mean return
        'goal': 'maximize'
    },
    'command': [
        '${env}',
        '${interpreter}',
        '${program}',
        '--lobs5_checkpoint', '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me/',
        '--replay_data_path', '/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc',
        '${args}'
    ],
    'parameters': {
        # --- Sweep Parameters ---
        'sigma': {
            'values': [0.005, 0.01, 0.02, 0.05]
        },
        'lr': {
            'values': [0.0001, 0.0005, 0.001, 0.005]
        },

        # --- Fixed Constraints ---
        'task_size': {
            'value': 50
        },
        'background_msgs_per_step': {
            'value': 50
        },
        'n_steps': {
            'value': 50
        },
        
        # --- Training Scale ---
        'pergpu_perturbations': {
            'value': 32
        },
        'n_epochs': {
            'value': 1000
        },
        'noiser': {
            'value': 'eggroll'
        },
        
        # --- Checkpoint & Data ---
        # NOTE: These must be passed as arguments or set via env vars in the agent script
        # But we can also set default values here if they are constant
    }
}

def main():
    parser = argparse.ArgumentParser(description='Setup WandB Sweep for ES')
    parser.add_argument('--project', type=str, default='es-lobs5-sweep', help='WandB project name')
    parser.add_argument('--entity', type=str, default='kang-oxford', help='WandB entity')
    args = parser.parse_args()

    print(f"Initializing sweep in {args.entity}/{args.project}...")
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=args.entity
    )
    
    print("\n" + "="*60)
    print(f"SWEEP CREATED SUCCESSFULLY!")
    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    print("="*60 + "\n")
    
    print("To launch agents, run:")
    print(f"  sbatch es_lobs5/scripts/run_sweep_agent.sh {sweep_id}")

if __name__ == '__main__':
    main()
