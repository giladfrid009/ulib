import torch
from torch import nn
from typing import Callable
import urllib.request
import json
import re

from ulib.utils.logging import create_logger
from ulib.utils.torch import get_device

from robustbench.utils import load_model as robust_load_model
from robustbench.data import get_preprocessing as robust_get_preprocessing
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset


logger = create_logger(__name__)


def _to_class_name(s: str) -> str:
    """
    Convert a string to a valid Python class name.

    Args:
        s: The string to convert.

    Returns:
        A valid Python class name.
    """
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)  # Replace invalid characters with underscores
    if not s[0].isalpha():  # Prepend an underscore if the first character isn't a letter
        s = "_" + s
    return s.title()


def _patch_class_name(object: object, name: str):
    """
    Patch the class name of an object.
    This is useful when you want to create a class with a dynamic name.

    Args:
        object: The object to patch.
        name: The new class name.
    """
    object.__class__ = type(_to_class_name(name), (object.__class__,), {})


class ModelInfo:
    def __init__(
        self,
        name: str | None = None,
        authors: str | None = None,
        architecture: str | None = None,
        clean_acc: float | None = None,
        **kwargs,
    ):
        self.name = name
        self.authors = authors
        self.architecture = architecture
        self.clean_acc = clean_acc
        self.__dict__.update(kwargs)


def robust_model_info(model_name: str, dataset: str = "imagenet", norm: str = "Linf") -> ModelInfo:
    url = f"https://raw.githubusercontent.com/RobustBench/robustbench/refs/heads/master/model_info/{dataset}/{norm}/{model_name}.json"
    try:
        with urllib.request.urlopen(url) as response:
            info = json.loads(response.read().decode("utf-8"))
            return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        return ModelInfo()


def load_robust_model(
    model_type: str,
    dataset: str,
    norm: str = "Linf",
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Callable]:
    dataset = dataset.lower()

    assert dataset in ["cifar10", "cifar100", "imagenet"], f"Unknown dataset: {dataset}"
    assert norm in ["Linf", "L2"], f"Unknown norm: {norm}"

    if device is None:
        device = get_device()

    ds_enum = BenchmarkDataset(dataset)
    norm_enum = ThreatModel(norm)

    model = robust_load_model(model_type, dataset=ds_enum, threat_model=norm_enum)
    model = model.eval().to(device)
    _patch_class_name(model, model_type)

    data_trans = robust_get_preprocessing(
        model_name=model_type,
        dataset=ds_enum,
        threat_model=norm_enum,
        preprocessing=None,
    )

    model_info = robust_model_info(model_type, dataset, norm)
    logger.info(f"Model        :  {type(model).__name__}")
    logger.info(f"Device       : {device}")
    logger.info(f"RB Name      : {model_info.name}")
    logger.info(f"RB Arch      : {model_info.architecture}")
    logger.info(f"RB Clean Acc : {model_info.clean_acc} ")

    return model, data_trans
