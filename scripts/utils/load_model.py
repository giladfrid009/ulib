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
    ALEXNET = "alextnet"
    RESNET50 = "resnet50"
    INCEPTION_V3 = "inception-v3"
    SALMAN_RESNET50 = "salman-resnet50"
    SINGH_VIT_S = "singh-vit-s"


SUPPORTED_MODELS = [e.value for e in ModelName]


def load_model(
    model_name: str,
    dataset_name: str,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Callable]:
    """
    Load a model by name.

    Args:
        model_name: The name of the model to load.

    Returns:
        A tuple containing the model and the data transformation function.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}.")

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset name: {dataset_name}.")

    if device is None:
        device = get_device()

    clear_memory()

    if model_name == ModelName.ALEXNET and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("alexnet", device=device)

    if model_name == ModelName.INCEPTION_V3 and dataset_name == DatasetName.IMAGENET:
        return load_torchvision_model("inception_v3", device=device)

    if model_name == ModelName.RESNET50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Standard_R50", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.SALMAN_RESNET50 and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Salman2020Do_R50", "imagenet", norm="Linf", device=device)

    if model_name == ModelName.SINGH_VIT_S and dataset_name == DatasetName.IMAGENET:
        return load_robust_model("Singh2023Revisiting_ViT-S-ConvStem", "imagenet", norm="Linf", device=device)

    raise ValueError(f"Unsupported configuration: model={model_name}, dataset={dataset_name}.")
