from re import S
from torchvision import datasets
from torch.utils.data import Subset
from typing import Callable
from enum import Enum

from ulib.data import TensorLoader, PartialImageFolder
from ulib.utils.logging import create_logger


logger = create_logger(__name__)


class DatasetName(str, Enum):
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


SUPPORTED_DATASETS = [e.value for e in DatasetName]


def load_dataset(
    name: str,
    transform: Callable,
    train_batch: int,
    eval_batch: int,
) -> tuple[TensorLoader, TensorLoader, TensorLoader]:
    """
    Loads a dataset by name and returns the training, validation, and test sets.

    Args:
        name (str): Name of the dataset to load. Must be one of the supported datasets.
        transform (Callable): A function/transform that takes in an PIL image and returns a transformed version.
        train_batch (int): Batch size for the training set.
        eval_batch (int): Batch size for the validation and test sets.

    Returns:
        tuple[TensorLoader, TensorLoader, TensorLoader]: A tuple containing the training, validation, and test data loaders.
    """

    if name == DatasetName.CIFAR10:
        ds_train = datasets.CIFAR10("/datasets", train=True, transform=transform)  # 50K samples
        ds_eval = datasets.CIFAR10("/datasets", train=False, transform=transform)  # 10K samples
        ds_test = Subset(ds_eval, list(range(0, len(ds_eval), 2)))  # 5K samples
        ds_eval = Subset(ds_eval, list(range(1, len(ds_eval), 2)))  # 5K samples

        dl_train = TensorLoader.from_dataset(ds_train, batch_size=train_batch, shuffle=True, loader_workers=0)
        dl_eval = TensorLoader.from_dataset(ds_eval, batch_size=eval_batch, shuffle=False, loader_workers=0)
        dl_test = TensorLoader.from_dataset(ds_test, batch_size=eval_batch, shuffle=False, loader_workers=0)
        return dl_train, dl_eval, dl_test

    if name == DatasetName.CIFAR100:
        ds_train = datasets.CIFAR100("/datasets", train=True, transform=transform)  # 50K samples
        ds_eval = datasets.CIFAR100("/datasets", train=False, transform=transform)  # 10K samples
        ds_test = Subset(ds_eval, list(range(0, len(ds_eval), 2)))  # 5K samples
        ds_eval = Subset(ds_eval, list(range(1, len(ds_eval), 2)))  # 5K samples

        dl_train = TensorLoader.from_dataset(ds_train, batch_size=train_batch, shuffle=True, loader_workers=0)
        dl_eval = TensorLoader.from_dataset(ds_eval, batch_size=eval_batch, shuffle=False, loader_workers=0)
        dl_test = TensorLoader.from_dataset(ds_test, batch_size=eval_batch, shuffle=False, loader_workers=0)
        return dl_train, dl_eval, dl_test

    if name == DatasetName.IMAGENET:
        ds_train = PartialImageFolder("/datasets/ImageNet/train", size=10000, transform=transform)
        ds_eval = PartialImageFolder("/datasets/ImageNet/val", size=20000, transform=transform)
        ds_test = Subset(ds_eval, list(range(0, len(ds_eval), 2)))  # 10K samples
        ds_eval = Subset(ds_eval, list(range(1, len(ds_eval), 2)))  # 10K samples

        dl_train = TensorLoader.from_dataset(ds_train, batch_size=train_batch, shuffle=True, loader_batch_size=1024)
        dl_eval = TensorLoader.from_dataset(ds_eval, batch_size=eval_batch, shuffle=False, loader_batch_size=1024)
        dl_test = TensorLoader.from_dataset(ds_test, batch_size=eval_batch, shuffle=False, loader_batch_size=1024)
        return dl_train, dl_eval, dl_test

    raise ValueError(f"Unsupported dataset name: {name}.")
