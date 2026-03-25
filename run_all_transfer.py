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

PROJECT_NAME = "VISION_COS_UAP"
SCRIPT_PATH = "scripts/evaluate.py"

# =============================================================================
# Experiment Setups
# =============================================================================

PERT_PATHS = [
    # COS UAP
    # # "logs-cos-uap/VISION_COS_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter1/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter2/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter3/best_pert.pt",
    # "logs-cos-uap/VISION_COS_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter4/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter1/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter2/best_pert.pt",
    # "logs-cos-uap/VISION_COS_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter3/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter4/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter1/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter2/best_pert.pt",
    # "logs-cos-uap/VISION_COS_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter3/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter4/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter1/best_pert.pt",
    # "logs-cos-uap/VISION_COS_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter2/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter3/best_pert.pt",
    # # "logs-cos-uap/VISION_COS_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter4/best_pert.pt",
    "logs-cos-uap/VISION_COS_UAP/resnet50/imagenet/L_inf/Eps_16/resnet50-std-iter1/best_pert.pt",
    "logs-cos-uap/VISION_COS_UAP/swin/imagenet/L_inf/Eps_16/swin-std-iter4/best_pert.pt",

    # DF UAP
    # # "logs-df-uap/VISION_DF_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-kappa_4-iter1/best_pert.pt",
    # "logs-df-uap/VISION_DF_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-kappa_4-iter2/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-kappa_4-iter3/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-kappa_4-iter4/best_pert.pt",
    # "logs-df-uap/VISION_DF_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-kappa_4-iter1/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-kappa_4-iter2/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-kappa_4-iter3/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-kappa_4-iter4/best_pert.pt",
    # "logs-df-uap/VISION_DF_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-kappa_4-iter1/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-kappa_4-iter2/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-kappa_4-iter3/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-kappa_4-iter4/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-kappa_4-iter1/best_pert.pt",
    # "logs-df-uap/VISION_DF_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-kappa_4-iter2/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-kappa_4-iter3/best_pert.pt",
    # # "logs-df-uap/VISION_DF_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-kappa_4-iter4/best_pert.pt",
    "logs-df-uap/VISION_DF_UAP/resnet50/imagenet/L_inf/Eps_16/resnet50-kappa_4-iter3/best_pert.pt",
    "logs-df-uap/VISION_DF_UAP/swin/imagenet/L_inf/Eps_16/swin-kappa_4-iter2/best_pert.pt",
    
    # FG UAP
    # "logs-fg-uap/VISION_FG_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter1/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter2/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter3/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-std-iter4/best_pert.pt",
    # "logs-fg-uap/VISION_FG_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter1/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter2/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter3/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-std-iter4/best_pert.pt",
    # "logs-fg-uap/VISION_FG_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter1/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter2/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter3/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-std-iter4/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter1/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter2/best_pert.pt",
    # # "logs-fg-uap/VISION_FG_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter3/best_pert.pt",
    # "logs-fg-uap/VISION_FG_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-std-iter4/best_pert.pt",
    "logs-fg-uap/VISION_FG_UAP/resnet50/imagenet/L_inf/Eps_16/resnet50-std-iter2/best_pert.pt",
    "logs-fg-uap/VISION_FG_UAP/swin/imagenet/L_inf/Eps_16/swin-std-iter1/best_pert.pt",
    
    # UAPGD
    # # "logs-uapgd/VISION_UAPGD/singh-vit/imagenet/L_inf/Eps_16/singh-vit-mod-iter1/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/singh-vit/imagenet/L_inf/Eps_16/singh-vit-mod-iter2/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/singh-vit/imagenet/L_inf/Eps_16/singh-vit-mod-iter3/best_pert.pt",
    # "logs-uapgd/VISION_UAPGD/singh-vit/imagenet/L_inf/Eps_16/singh-vit-mod-iter4/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-mod-iter1/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-mod-iter2/best_pert.pt",
    # "logs-uapgd/VISION_UAPGD/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-mod-iter3/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-mod-iter4/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-swin/imagenet/L_inf/Eps_16/liu-swin-mod-iter1/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-swin/imagenet/L_inf/Eps_16/liu-swin-mod-iter2/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/liu-swin/imagenet/L_inf/Eps_16/liu-swin-mod-iter3/best_pert.pt",
    # "logs-uapgd/VISION_UAPGD/liu-swin/imagenet/L_inf/Eps_16/liu-swin-mod-iter4/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-mod-iter1/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-mod-iter2/best_pert.pt",
    # "logs-uapgd/VISION_UAPGD/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-mod-iter3/best_pert.pt",
    # # "logs-uapgd/VISION_UAPGD/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-mod-iter4/best_pert.pt",
    "logs-uapgd/VISION_UAPGD/resnet50/imagenet/L_inf/Eps_16/resnet50-mod-iter3/best_pert.pt",
    "logs-uapgd/VISION_UAPGD/swin/imagenet/L_inf/Eps_16/swin-mod-iter1/best_pert.pt",
    
    # IML
    # # "logs-iml/VISION_IML_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-fast-iter1/best_pert.pt",
    # "logs-iml/VISION_IML_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-fast-iter2/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-fast-iter3/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/singh-vit/imagenet/L_inf/Eps_16/singh-vit-fast-iter4/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-fast-iter1/best_pert.pt",
    # "logs-iml/VISION_IML_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-fast-iter2/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-fast-iter3/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-convnext/imagenet/L_inf/Eps_16/liu-convnext-fast-iter4/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-fast-iter1/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-fast-iter2/best_pert.pt",
    # "logs-iml/VISION_IML_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-fast-iter3/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/liu-swin/imagenet/L_inf/Eps_16/liu-swin-fast-iter4/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-fast-iter1/best_pert.pt",
    # "logs-iml/VISION_IML_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-fast-iter2/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-fast-iter3/best_pert.pt",
    # # "logs-iml/VISION_IML_UAP/salman-resnet50/imagenet/L_inf/Eps_16/salman-resnet50-fast-iter4/best_pert.pt",
    "logs-iml/VISION_IML_UAP/resnet50/imagenet/L_inf/Eps_16/resnet50-std-iter4/best_pert.pt",
    "logs-iml/VISION_IML_UAP/swin/imagenet/L_inf/Eps_16/swin-std-iter1/best_pert.pt",
]


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consolidated Cosine UAP experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], help="List of model keys to run, or 'all'")

    args = parser.parse_args()
    
    for pert_path in PERT_PATHS:
        cmd = [
            "python", 
            SCRIPT_PATH,
            pert_path,
            "--overwrite",
        ]
        
        if "all" not in args.models:
            cmd += ["--models"] + args.models
        
        try:
            print("Executing:", " ".join(cmd))
            collect_garbage()
            subprocess.run(cmd, shell=False, check=True)
        except KeyboardInterrupt:
            print("\n🚨 Execution Interrupted by User")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during execution of {pert_path}: {e}")
            # Continue to next experiment/model
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
