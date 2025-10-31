import torch
from torch import nn
from typing import Callable
from torchvision import transforms, models
from collections import OrderedDict
import re

from ulib.utils.logging import create_logger
from ulib.utils.torch import get_device


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


def load_torchvision_model(
    model_type: str,
    device: str | torch.device | None = None,
    **model_kwargs,
) -> tuple[nn.Module, Callable]:
    if device is None:
        device = get_device()

    weights_enum = models.get_model_weights(model_type)
    weights: models.Weights = weights_enum.DEFAULT  # type: ignore
    orig_model = models.get_model(name=model_type, weights=weights, progress=True, **model_kwargs)

    orig_trans = weights.transforms()
    data_trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                size=orig_trans.resize_size,
                interpolation=orig_trans.interpolation,
                antialias=orig_trans.antialias if orig_trans.antialias is not None else False,
            ),
            transforms.CenterCrop(size=orig_trans.crop_size),
            transforms.ConvertImageDtype(dtype=torch.float),
        ]
    )

    layers = OrderedDict(
        [
            ("normalize", transforms.Normalize(mean=orig_trans.mean, std=orig_trans.std)),
            ("model", orig_model),
        ]
    )

    model = nn.Sequential(layers)  # type: ignore
    model = model.eval().to(device)
    _patch_class_name(model, model_type)

    logger.info(f"Model     : {orig_model.__class__.__name__}")
    logger.info(f"Device    : {device}")
    logger.info(f"Weight    : {weights_enum.DEFAULT.name}")  # type: ignore
    logger.info(f"Metrics   : {weights.meta.get('_metrics', None)}")

    return model, data_trans
