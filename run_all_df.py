import subprocess
import argparse
import sys
import copy
from typing import Dict, List, Any
import torch
import gc

# =============================================================================
# Global Constants & Configuration
# =============================================================================

PROJECT_NAME = "DF_UAP_CV"
SCRIPT_PATH = "scripts/run_df_uap.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs/run_all_df",
    "dataset": "imagenet",
    "test_datasets": [],
    "evaluator": "simple",
    "train_batch": 32,
    "eval_batch": 64,
    "max_epochs": 10,
    "epsilon": 16,
    "norm": float("inf"),
    "use_amp": "true",
}

# =============================================================================
# Experiment Setups
# =============================================================================

EXP_SETUP_STD = dict(
    lr=0.005,
    kappa=0.0,
)

EXP_SETUP_HIGH_KAPPA = dict(
    lr=0.005,
    kappa=10.0,
)

# =============================================================================
# Model Registry
# =============================================================================

MODELS = {
    "resnet50": {
        "nick": "R50",
        "experiments": {"std": EXP_SETUP_STD, "h_kappa": EXP_SETUP_HIGH_KAPPA},
        # CLI Arguments
        "model": "resnet50",
    },
    "salman_r50": {
        "nick": "SalmanR50",
        "experiments": {"std": EXP_SETUP_STD},
        # CLI Arguments
        "model": "salman-resnet50",
    },
    "vits": {
        "nick": "ViT-S",
        "experiments": {"std": EXP_SETUP_STD},
        # CLI Arguments
        "model": "singh-vit-s",
    },
}

# =============================================================================
# Helper Functions
# =============================================================================


def build_command(run_config: Dict[str, Any], script_path: str, run_name: str, project_name: str) -> List[str]:
    """Constructs the subprocess command list from configuration dictionary."""
    cmd_args = [
        "python",
        script_path,
        "--run_name",
        run_name,
        "--project_name",
        project_name,
    ]

    for arg_name, arg_value in run_config.items():
        if arg_value is None:
            continue

        # Skip empty lists to avoid argparse errors with nargs='+'
        if isinstance(arg_value, list) and len(arg_value) == 0:
            continue

        cmd_args.append(f"--{arg_name}")
        # Handle lists (repeated arguments or specific formatting based on implementation)
        # Assuming argparse nargs='+' style inputs
        if isinstance(arg_value, list):
            cmd_args.extend(map(str, arg_value))
        else:
            cmd_args.append(str(arg_value))

    return cmd_args


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def run_experiment(model_key: str, iteration: int, total_iters: int, name_suffix: str):
    """Executes experiments for a single model configuration."""

    # 1. Retrieve Model Configuration
    # Safe copy to avoid side effects
    if model_key not in MODELS:
        print(f"!! Model {model_key} not found in registry.")
        return

    model_conf = copy.deepcopy(MODELS[model_key])

    # Extract special control fields
    model_nick = model_conf.pop("nick", model_key)
    model_experiments = model_conf.pop("experiments", {})

    print(f"\n>> Processing Model: {model_key} ({model_nick})")

    if not model_experiments:
        print(f"!! No experiments defined for {model_key}. Skipping.")
        return

    # 2. Iterate over experiments defined for this model
    for exp_setup_name, exp_params in model_experiments.items():
        full_run_name = f"{model_nick}-{exp_setup_name}{name_suffix}-iter{iteration}"

        print("=" * 60)
        print(f"▶️ Running Experiment: {full_run_name} ({iteration}/{total_iters})")
        print("=" * 60)

        # 3. Construct Run Parameters Strategy:
        #    Global Defaults
        #    -> MERGED WITH -> Model Params
        #    -> MERGED WITH -> Experiment Params

        run_params = copy.deepcopy(DEFAULT_RUN_ARGS)

        # Add all remaining model config items as CLI arguments
        run_params.update(model_conf)

        # Add experiment-specific overrides
        run_params.update(exp_params)

        # 4. Build and Execute Command
        cmd = build_command(run_config=run_params, script_path=SCRIPT_PATH, run_name=full_run_name, project_name=PROJECT_NAME)

        try:
            print("Executing:", " ".join(cmd))
            collect_garbage()
            subprocess.run(cmd, shell=False, check=True)
        except KeyboardInterrupt:
            print("\n🚨 Execution Interrupted by User")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during execution of {full_run_name}: {e}")
            # Continue to next experiment/model
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consolidated DF-UAP experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting DF-UAP Runs for models: {target_models}")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            run_experiment(model_key, iteration, args.iters, args.name_suffix)
