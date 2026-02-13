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

PROJECT_NAME = "VISION_IML_UAP_ABLATIONS"
SCRIPT_PATH = "scripts/run_iml_uap.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs-iml-ablations",
    "dataset": "imagenet",
    "evaluator": "simple",
    "train_batch": 128,
    "eval_batch": 128,
    "max_epochs": 30,
    "max_time": 90,
    "patience": 10,
    "epsilon": 16,
    "norm": float("inf"),
    "use_amp": "true",
}

# =============================================================================
# Experiment Setups
# =============================================================================

EXP_SETUP_STD = dict(
    lr=0.1,
    skip_fooled="true",
    skip_failed="true",
)

EXP_SETUP_CW = dict(
    lr=0.1,
    skip_fooled="true",
    skip_failed="true",
    inner_attack="cw",
    inner_epsilon=2.0,
)

EXP_SETUP_FGSM = dict(
    lr=0.1,
    skip_fooled="true",
    skip_failed="true",
    inner_attack="fgsm",
)

EXP_SETUP_FFGSM = dict(
    lr=0.1,
    skip_fooled="true",
    skip_failed="true",
    inner_attack="ffgsm",
)

EXP_SETUP_PGD = dict(
    lr=0.1,
    skip_fooled="true",
    skip_failed="true",
    inner_attack="pgd",
)

EXP_SETUP_PGD_SLOW = dict(
    lr=0.005,
    skip_fooled="true",
    skip_failed="true",
    inner_attack="pgd",
)

# =============================================================================
# Model Registry
# =============================================================================

MODELS = {
    # standard models
    # "vit-b": {
    #     "nick": "vit-b",
    #     "experiments": {"std": EXP_SETUP_STD},
    #     # CLI Arguments
    #     "model": "vit-b",
    #     "layers": ["model.encoder"],
    #     "inner_attack": "ffgsm",
    # },
    "convnext": {
        "nick": "convnext",
        "experiments": {
            "cw": EXP_SETUP_CW,
            # "fgsm": EXP_SETUP_FGSM,
            # "ffgsm": EXP_SETUP_FFGSM,
            # "pgd": EXP_SETUP_PGD,
        },
        # CLI Arguments
        "model": "convnext",
        "layers": ["model.classifier.1"],
    },
    "swin": {
        "nick": "swin",
        "experiments": {
            "cw": EXP_SETUP_CW,
            # "fgsm": EXP_SETUP_FGSM,
            # "ffgsm": EXP_SETUP_FFGSM,
            # "pgd": EXP_SETUP_PGD,
            # "pgd-multi-mlp": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.mlp", "model.features.5.3.mlp", "model.features.3.1.mlp"]),
            # "pgd-multi-attn": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.attn", "model.features.5.3.attn", "model.features.3.1.attn"]),
            # "pgd-multi-patch-merge": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.2", "model.features.4", "model.features.6"]),
            # "pgd-multi-attn-qkv": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.attn.qkv", "model.features.5.3.attn.qkv", "model.features.3.1.attn.qkv"]),
            # "pgd-multi-attn-proj": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.attn.proj", "model.features.5.3.attn.proj", "model.features.3.1.attn.proj"]),
            # "pgd-multi-mlp-norm": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.norm2", "model.features.5.3.norm2", "model.features.3.1.norm2"]),
            # "pgd-multi-attn-norm": dict(**EXP_SETUP_PGD_SLOW, layers=["model.flatten", "model.features.7.1.norm1", "model.features.5.3.norm1", "model.features.3.1.norm1"]),
            # "pgd_slow": EXP_SETUP_PGD_SLOW,
        },
        # CLI Arguments
        "model": "swin",
        "layers": ["model.flatten"],
    },
    # robust models
    # "wong-resnet50-1": {
    #     "nick": "wong-resnet50",
    #     "experiments": {
    #         "cw": EXP_SETUP_CW,
    #         # "fgsm": EXP_SETUP_FGSM,
    #         # "ffgsm": EXP_SETUP_FFGSM,
    #         # "pgd": EXP_SETUP_PGD,
    #         # "ffgsm-multi-layer": dict(**EXP_SETUP_FFGSM, layers=["model.avgpool", "model.layer1", "model.layer2", "model.layer3"]),
    #         # "ffgsm-multi-downsample": dict(**EXP_SETUP_FFGSM, layers=["model.avgpool", "model.layer1.0.downsample", "model.layer2.0.downsample", "model.layer3.0.downsample", "model.layer4.0.downsample"]),
    #         # "ffgsm-multi-conv": dict(**EXP_SETUP_FFGSM, layers=["model.avgpool", "model.layer1.2.conv2", "model.layer2.2.conv2", "model.layer3.2.conv2", "model.layer4.1.conv2"]),
    #         # "ffgsm-multi-bn": dict(**EXP_SETUP_FFGSM, layers=["model.avgpool", "model.layer1.2.bn2", "model.layer2.2.bn2", "model.layer3.2.bn2", "model.layer4.1.bn2"]),
    #         },
    #     # CLI Arguments
    #     "model": "wong-resnet50",
    #     "layers": ["model.avgpool"],
    # },
    # "wong-resnet50-2": {
    #     "nick": "wong-resnet50",
    #     "experiments": {
    #         # "pgd-multi-layer": dict(**EXP_SETUP_PGD, layers=["model.avgpool", "model.layer1", "model.layer2", "model.layer3"]),
    #         # "pgd-multi-downsample": dict(**EXP_SETUP_PGD, layers=["model.avgpool", "model.layer1.0.downsample", "model.layer2.0.downsample", "model.layer3.0.downsample", "model.layer4.0.downsample"]),
    #         # "pgd-multi-conv": dict(**EXP_SETUP_PGD, layers=["model.avgpool", "model.layer1.2.conv2", "model.layer2.2.conv2", "model.layer3.2.conv2", "model.layer4.1.conv2"]),
    #         # "pgd-multi-bn": dict(**EXP_SETUP_PGD, layers=["model.avgpool", "model.layer1.2.bn2", "model.layer2.2.bn2", "model.layer3.2.bn2", "model.layer4.1.bn2"]),
    #         "cw": EXP_SETUP_CW,
    #         "cw-multi-layer": dict(**EXP_SETUP_CW, layers=["model.avgpool", "model.layer1", "model.layer2", "model.layer3"]),
    #         "cw-multi-downsample": dict(**EXP_SETUP_CW, layers=["model.avgpool", "model.layer1.0.downsample", "model.layer2.0.downsample", "model.layer3.0.downsample", "model.layer4.0.downsample"]),
    #         "cw-multi-conv": dict(**EXP_SETUP_CW, layers=["model.avgpool", "model.layer1.2.conv2", "model.layer2.2.conv2", "model.layer3.2.conv2", "model.layer4.1.conv2"]),
    #         "cw-multi-bn": dict(**EXP_SETUP_CW, layers=["model.avgpool", "model.layer1.2.bn2", "model.layer2.2.bn2", "model.layer3.2.bn2", "model.layer4.1.bn2"]),
    #     },
    #     # CLI Arguments
    #     "model": "wong-resnet50",
    #     "inner_epsilon": 2.0,
    #     "layers": ["model.avgpool"],
    # },
    "salman-resnet50": {
        "nick": "salman-resnet50",
        "experiments": {
            "cw": EXP_SETUP_CW,
        },
        # CLI Arguments
        "model": "salman-resnet50",
        "layers": ["model.avgpool"],
    },
    "singh-vit": {
        "nick": "singh-vit",
        "experiments": {
            "cw": EXP_SETUP_CW,
        },
        # CLI Arguments
        "model": "singh-vit",
        "layers": ["model.head_drop"],
    },
    "liu-convnext": {
        "nick": "liu-convnext",
        "experiments": {
            "cw": EXP_SETUP_CW,
            # "fgsm": EXP_SETUP_FGSM,
            # "ffgsm": EXP_SETUP_FFGSM,
            # "pgd": EXP_SETUP_PGD,
            # "fgsm-multi-block": dict(**EXP_SETUP_FGSM, layers=["model.head.drop", "model.stages.2.blocks.10", "model.stages.2.blocks.20", "model.stages.2.blocks.26", "model.stages.3.blocks.1"]),
            # "fgsm-multi-mlp": dict(**EXP_SETUP_FGSM, layers=["model.head.drop", "model.stages.2.blocks.10.mlp", "model.stages.2.blocks.20.mlp", "model.stages.2.blocks.26.mlp", "model.stages.3.blocks.1.mlp"]),
            # "fgsm-multi-mlp-norm": dict(**EXP_SETUP_FGSM, layers=["model.head.drop", "model.stages.2.blocks.10.mlp.norm", "model.stages.2.blocks.20.mlp.norm", "model.stages.2.blocks.26.mlp.norm", "model.stages.3.blocks.1.mlp.norm"]),
            # "fgsm-multi-mlp-conv": dict(**EXP_SETUP_FGSM, layers=["model.head.drop", "model.stages.2.blocks.10.conv_dw", "model.stages.2.blocks.20.conv_dw", "model.stages.2.blocks.26.conv_dw", "model.stages.3.blocks.1.conv_dw"]),
            # "fgsm-multi-downsample": dict(**EXP_SETUP_FGSM, layers=["model.head.drop", "model.stages.1.downsample", "model.stages.2.downsample", "model.stages.3.downsample"]),
        },
        # CLI Arguments
        "model": "liu-convnext",
        "layers": ["model.head.drop"],
    },
    "liu-swin": {
        "nick": "liu-swin",
        "experiments": {
            "cw": EXP_SETUP_CW,
            # "fgsm": EXP_SETUP_FGSM,
            # "ffgsm": EXP_SETUP_FFGSM,
            # "pgd": EXP_SETUP_PGD,
            # "ffgsm-multi-block": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5", "model.layers.2.blocks.12", "model.layers.2.blocks.17", "model.layers.3.blocks.0"]),
            # "ffgsm-multi-mlp": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.mlp", "model.layers.2.blocks.12.mlp", "model.layers.2.blocks.17.mlp", "model.layers.3.blocks.0.mlp"]),
            # "ffgsm-multi-mlp-input": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.norm2", "model.layers.2.blocks.12.norm2", "model.layers.2.blocks.17.norm2", "model.layers.3.blocks.0.norm2"]),
            # "ffgsm-multi-attn-norm": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.norm1", "model.layers.2.blocks.12.norm1", "model.layers.2.blocks.17.norm1", "model.layers.3.blocks.0.norm1"]),
            # "ffgsm-multi-attn-qkv": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.attn.qkv", "model.layers.2.blocks.12.attn.qkv", "model.layers.2.blocks.17.attn.qkv", "model.layers.3.blocks.0.attn.qkv"]),
            # "ffgsm-multi-attn-proj": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.attn.proj", "model.layers.2.blocks.12.attn.proj", "model.layers.2.blocks.17.attn.proj", "model.layers.3.blocks.0.attn.proj"]),
            # "ffgsm-multi-mlp-norm": dict(**EXP_SETUP_FFGSM, layers=["model.head.drop", "model.layers.2.blocks.5.mlp.norm", "model.layers.2.blocks.12.mlp.norm", "model.layers.2.blocks.17.mlp.norm", "model.layers.3.blocks.0.mlp.norm"]),
        },
        # CLI Arguments
        "model": "liu-swin",
        "layers": ["model.head.drop"],
    },
    # "peng-wrn50": {
    #     "nick": "peng-wrn50",
    #     "experiments": {"std": EXP_SETUP_STD},
    #     # CLI Arguments
    #     "model": "peng-wrn50",
    #     "layers": [],
    #     "inner_attack": "ffgsm",
    # },
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
    parser = argparse.ArgumentParser(description="Run consolidated IML experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting IML Runs for models: {target_models}")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            run_experiment(model_key, iteration, args.iters, args.name_suffix)
