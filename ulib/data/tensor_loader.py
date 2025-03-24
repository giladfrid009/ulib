from __future__ import annotations
import random
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Iterator, Callable
from tqdm.auto import tqdm


class TensorLoader:
    """
    This class provides an efficient way to iterate over tensor data in batches, with optional
    shuffling and transformation capabilities. It's designed to be significantly faster than
    PyTorch's DataLoader when working with tensor datasets.
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Args:
            data (torch.Tensor): The input tensor data.
            targets (torch.Tensor): The target tensor data.
            batch_size (int): Size of batches to yield when iterating.
            shuffle (bool): Whether to shuffle data when iterating.
            drop_last (bool): Whether to drop the last incomplete batch.
            transform (Callable[[torch.Tensor], torch.Tensor], optional): A function/transform that takes in the input and transforms it.
            target_transform (Callable[[torch.Tensor], torch.Tensor], optional): A function/transform that takes in the target and transforms it.
        """
        if len(data) != len(targets):
            raise ValueError("data and targets must have the same length.")

        if drop_last:
            total = len(data) - len(data) % batch_size
            data = data[:total]
            targets = targets[:total]

        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.target_transform = target_transform

    @property
    def total(self) -> int:
        return len(self.data)

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        size: int | None = None,
        batch_size: int = 256,
        loader_batch_size: int = 512,
        loader_workers: int = 5,
        **kwargs,
    ) -> TensorLoader:
        """
        Create a TensorLoader instance from a PyTorch Dataset by loading the data into memory as tensors.

        Args:
            ds (Dataset): The PyTorch Dataset to load from.
            size (int, optional): Number of samples to load from the dataset. If None, load the entire dataset.
            batch_size (int): Size of batches to yield when iterating.
            loader_batch_size (int): Batch size to use for data extraction.
            loader_workers (int): Number of workers to use for data extraction.
            **kwargs: Additional arguments to pass to the TensorLoader constructor.
        """
        if size is not None and size < len(ds):
            indices = torch.tensor(random.sample(range(len(ds)), k=size), dtype=torch.int64)
            ds = Subset(ds, indices)

        data_list = []
        targets_list = []

        dl = DataLoader(ds, batch_size=loader_batch_size, num_workers=loader_workers, shuffle=False)
        
        # NOTE: tqdm shouldnt directly wrap the DataLoader since it throws 
        # weird exceptions when using tqdm.auto and num_workers > 0
        with tqdm(desc="Extracting Samples", leave=False, total=len(dl)) as pbar:
            for sample in dl:
                data_list.append(sample[0])
                targets_list.append(sample[1])
                pbar.update()

        data = torch.cat(data_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        if size is not None:
            data = data[:size]
            targets = targets[:size]

        return cls(data, targets, batch_size=batch_size, **kwargs)

    def __len__(self) -> int:
        return self.total // self.batch_size + int(self.total % self.batch_size != 0)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = torch.randperm(self.total) if self.shuffle else torch.arange(self.total)
        for i in range(0, self.total, self.batch_size):
            idx = indices[i : i + self.batch_size]

            data = self.data[idx]
            targets = self.targets[idx]

            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                targets = self.target_transform(targets)

            yield data, targets
