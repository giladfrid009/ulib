from torchvision import transforms, datasets
from typing import Callable
from ulib.data import TensorLoader, PartialImageFolder


def load_cifar10(transform: Callable = transforms.ToTensor(), batch_size: int = 256) -> tuple[TensorLoader, TensorLoader]:
    ds_train = datasets.CIFAR10(root="/datasets", train=True, download=False, transform=transform)
    ds_eval = datasets.CIFAR10(root="/datasets", train=False, download=False, transform=transform)
    dl_train = TensorLoader.from_dataset(ds_train, size=50000, batch_size=batch_size, shuffle=True, loader_workers=0)
    dl_eval = TensorLoader.from_dataset(ds_eval, size=10000, batch_size=batch_size, shuffle=False, loader_workers=0)
    return dl_train, dl_eval


def load_cifar100(transform: Callable = transforms.ToTensor(), batch_size: int = 256) -> tuple[TensorLoader, TensorLoader]:
    ds_train = datasets.CIFAR100(root="/datasets", train=True, download=False, transform=transform)
    ds_eval = datasets.CIFAR100(root="/datasets", train=False, download=False, transform=transform)
    dl_train = TensorLoader.from_dataset(ds_train, size=50000, batch_size=batch_size, shuffle=True, loader_workers=0)
    dl_eval = TensorLoader.from_dataset(ds_eval, size=10000, batch_size=batch_size, shuffle=False, loader_workers=0)
    return dl_train, dl_eval


def load_imagenet(transform: Callable, batch_size: int = 256) -> tuple[TensorLoader, TensorLoader]:
    ds_train = PartialImageFolder(root="/datasets/ImageNet/train", size=50000, allow_empty=False, transform=transform)
    ds_eval = PartialImageFolder(root="/datasets/ImageNet/val", size=10000, allow_empty=False, transform=transform)
    dl_train = TensorLoader.from_dataset(ds_train, batch_size=batch_size, shuffle=True, loader_batch_size=1024, loader_workers=5)
    dl_eval = TensorLoader.from_dataset(ds_eval, batch_size=batch_size, shuffle=False, loader_batch_size=1024, loader_workers=5)
    return dl_train, dl_eval
