from re import M
import torch
from torch import nn
from typing import Callable
from enum import Enum

from ulib.utils.logging import create_logger
from ulib.utils.torch import clear_memory, get_device

from scripts.utils.load_model_torchvision import load_torchvision_model
from scripts.utils.load_model_robustbench import load_robust_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, DatasetName


logger = create_logger(__name__)


class ModelName(str, Enum):
    # standard models
    ALEXNET = "alextnet"
    RESNET50 = "resnet50"
    INCEPTION_V3 = "inception-v3"
    VIT = "vit-b"
    CONV_NEXT = "convnext"
    SWIN_V2 = "swin"
    # robust models
    WONG_RESNET50 = "wong-resnet50"
    SALMAN_RESNET50 = "salman-resnet50"
    SINGH_VIT = "singh-vit"
    MO_VIT = "mo-vit"
    MO_SWIN = "mo-swin"
    LIU_CONV_NEXT = "liu-convnext"
    LIU_SWIN = "liu-swin"
    DEB_XCIT = "deb-xcit"
    PENG_WRN50 = "peng-wrn50"


SUPPORTED_MODELS = [e.value for e in ModelName]


def load_model(
    model_name: str | ModelName,
    dataset_name: str | DatasetName,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Callable]:
    """
    Load a model by name.

    Args:
        model_name: The name of the model to load.

    Returns:
        A tuple containing the model and the data transformation function.
    """
    if isinstance(model_name, ModelName):
        model_name = model_name.value
        
    if isinstance(dataset_name, DatasetName):
        dataset_name = dataset_name.value
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}.")

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset name: {dataset_name}.")

    if device is None:
        device = get_device()

    clear_memory()

    # standard models

    if model_name == ModelName.ALEXNET and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("alexnet", device=device)

    if model_name == ModelName.RESNET50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Standard_R50", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.INCEPTION_V3 and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("inception_v3", device=device)

    if model_name == ModelName.VIT and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("vit_b_16", device=device)

    if model_name == ModelName.CONV_NEXT and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("convnext_tiny", device=device)

    if model_name == ModelName.SWIN_V2 and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("swin_v2_t", device=device)

    # robust models

    if model_name == ModelName.WONG_RESNET50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Wong2020Fast", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.SALMAN_RESNET50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Salman2020Do_R50", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.SINGH_VIT and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Singh2023Revisiting_ViT-S-ConvStem", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.MO_VIT and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Mo2022When_ViT-B", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.MO_SWIN and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Mo2022When_Swin-B", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.LIU_CONV_NEXT and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Liu2023Comprehensive_ConvNeXt-B", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.LIU_SWIN and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Liu2023Comprehensive_Swin-B", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.DEB_XCIT and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Debenedetti2022Light_XCiT-S12", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.PENG_WRN50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Peng2023Robust", "imagenet", norm="Linf", device=device)

    raise ValueError(f"Unsupported configuration: model={model_name}, dataset={dataset_name}.")
