from torch import nn

from robustbench.utils import load_model as robust_load_model
from robustbench.data import get_preprocessing as robust_get_preprocessing
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset

from ulib import utils, eval
from ulib.data import TensorLoader
from notebooks.datasets import load_cifar10, load_cifar100, load_imagenet
from notebooks.experiment_utils import patch_class_name

import urllib.request
import json


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


def get_info(model_name: str, dataset: str = "imagenet", norm: str = "Linf") -> ModelInfo:
    url = f"https://raw.githubusercontent.com/RobustBench/robustbench/refs/heads/master/model_info/{dataset}/{norm}/{model_name}.json"
    try:
        with urllib.request.urlopen(url) as response:
            info = json.loads(response.read().decode("utf-8"))
            return ModelInfo(**info)
    except Exception as e:
        print(f"Error loading model info: {e}")
        return ModelInfo()


def load_robust_experiment(
    model_type: str,
    dataset: str,
    norm: str = "Linf",
    batch_size: int = 256,
    silent: bool = False,
) -> tuple[nn.Module, TensorLoader, TensorLoader]:

    dataset = dataset.lower()

    assert dataset in ["cifar10", "cifar100", "imagenet"], f"Unknown dataset: {dataset}"
    assert norm in ["Linf", "L2"], f"Unknown norm: {norm}"

    utils.clear_memory()
    device = utils.get_device()
    ds_enum = BenchmarkDataset(dataset)
    norm_emum = ThreatModel(norm)

    model = robust_load_model(model_type, dataset=ds_enum, threat_model=norm_emum)
    model = model.eval().to(device)
    patch_class_name(model, model_type)
    
    data_trans = robust_get_preprocessing(
        model_name=model_type,
        dataset=ds_enum,
        threat_model=norm_emum,
        preprocessing=None,
    )

    if ds_enum == BenchmarkDataset.cifar_10:
        dl_train, dl_eval = load_cifar10(transform=data_trans, batch_size=batch_size)
    elif ds_enum == BenchmarkDataset.cifar_100:
        dl_train, dl_eval = load_cifar100(transform=data_trans, batch_size=batch_size)
    elif ds_enum == BenchmarkDataset.imagenet:
        dl_train, dl_eval = load_imagenet(transform=data_trans, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not silent:
        clean_train_acc = eval.accuracy(model, dl_train, silent=False)
        clean_eval_acc = eval.accuracy(model, dl_eval, silent=False)
        model_info = get_info(model_type, dataset, norm)
        print(f"Model        :  {model.__class__.__name__}")
        print(f"Device       : {device}")
        print(f"Eval Acc     : {round(clean_eval_acc * 100, 2)}")
        print(f"Train Acc    : {round(clean_train_acc * 100, 2)}")
        print(f"RB Name      : {model_info.name}")
        print(f"RB Arch      : {model_info.architecture}")
        print(f"RB Clean Acc : {model_info.clean_acc} ")

    return model, dl_train, dl_eval
