import torch
from torch import nn
from torchvision import transforms, models
from collections import OrderedDict

from notebooks.datasets import load_imagenet
from notebooks.experiment_utils import patch_class_name
from ulib import utils, eval
from ulib.data import TensorLoader


def load_torchvision_experiment(
    model_type: str,
    batch_size: int = 256,
    silent: bool = False,
    **model_kwargs,
) -> tuple[nn.Module, TensorLoader, TensorLoader]:
    utils.clear_memory()
    device = utils.get_device()

    weights_enum = models.get_model_weights(model_type)
    weights: models.Weights = weights_enum.DEFAULT
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
    patch_class_name(model, model_type)

    dl_train, dl_eval = load_imagenet(transform=data_trans, batch_size=batch_size)

    if not silent:
        clean_train_acc = eval.accuracy(model, dl_train, silent=False)
        clean_eval_acc = eval.accuracy(model, dl_eval, silent=False)
        print(f"Model     : {orig_model.__class__.__name__}")
        print(f"Device    : {device}")
        print(f"Eval Acc  : {round(clean_eval_acc * 100, 2)}")
        print(f"Train Acc : {round(clean_train_acc * 100, 2)}")
        print(f"Weight    : {weights_enum.DEFAULT.name}")
        print(f"Metrics   :", weights.meta.get("_metrics", None))

    return model, dl_train, dl_eval
