from __future__ import annotations
import random
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Iterator, Callable
from tqdm.auto import tqdm


class TensorLoader:
    """
    Efficient iterator over multiple tensor datasets in batches with optional shuffling and transformations.
    Designed to be faster than PyTorch's DataLoader for tensor datasets.
    """

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        transforms: tuple[Callable[[torch.Tensor], torch.Tensor] | None, ...] | None = None,
        tensor_names: tuple[str, ...] | None = None,
    ):
        """
        Args:
            tensors (tuple[torch.Tensor, ...]): Tuple of tensor data.
            batch_size (int): Size of batches to yield when iterating.
            shuffle (bool): Whether to shuffle data when iterating.
            drop_last (bool): Whether to drop the last incomplete batch.
            transforms (tuple[Callable[[torch.Tensor], torch.Tensor] | None, ...], optional):
                Tuple of functions/transforms for each tensor. None means no transform.
            tensor_names (tuple[str, ...], optional): Names for each tensor for better readability.
        """
        if not tensors:
            raise ValueError("tensors tuple cannot be empty")

        length = len(tensors[0])
        if not all(len(t) == length for t in tensors):
            raise ValueError("All tensors must have the same length")

        if not transforms:
            transforms = tuple(lambda x: x for _ in tensors)

        transforms = tuple(t if t is not None else lambda x: x for t in transforms)

        if len(transforms) != len(tensors):
            raise ValueError("Number of transforms must match number of tensors")

        if tensor_names is not None and len(tensor_names) != len(tensors):
            raise ValueError("Number of tensor names must match number of tensors")

        if drop_last:
            total = length - length % batch_size
            tensors = tuple(t[:total] for t in tensors)

        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transforms = transforms
        self.tensor_names = tensor_names or tuple(f"tensor_{i}" for i in range(len(tensors)))

    @property
    def total(self) -> int:
        """Total number of samples."""
        return len(self.tensors[0])

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the tuple."""
        return len(self.tensors)

    def get_tensor(self, index: int | str) -> torch.Tensor:
        """Get a specific tensor by index or name."""
        if isinstance(index, str):
            try:
                index = self.tensor_names.index(index)
            except ValueError:
                raise ValueError(f"Tensor name '{index}' not found")
        return self.tensors[index]

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
        Create a TensorLoader from a PyTorch Dataset.

        Args:
            ds (Dataset): Dataset returning a tuple of tensors.
            size (int, optional): Number of samples to load. If None, load all.
            batch_size (int): Batch size for iteration.
            loader_batch_size (int): Batch size for data extraction.
            loader_workers (int): Number of workers for data extraction.
            **kwargs: Additional arguments for TensorLoader constructor.
        """
        if size is not None and size < len(ds):
            indices = torch.tensor(random.sample(range(len(ds)), k=size), dtype=torch.int64)
            ds = Subset(ds, indices)

        tensor_lists = []
        dl = DataLoader(ds, batch_size=loader_batch_size, num_workers=loader_workers, shuffle=False)

        # NOTE: tqdm shouldnt directly wrap the DataLoader since it throws 
        # weird exceptions when using tqdm.auto and num_workers > 0
        with tqdm(desc="Extracting Samples", leave=False, total=len(dl)) as pbar:
            for batch in dl:
                if not tensor_lists:
                    tensor_lists = [[] for _ in range(len(batch))]
                for i, tensor in enumerate(batch):
                    tensor_lists[i].append(tensor)
                pbar.update()

        tensors = tuple(torch.cat(t_list, dim=0) for t_list in tensor_lists)

        if size is not None:
            tensors = tuple(t[:size] for t in tensors)

        return cls(tensors, batch_size=batch_size, **kwargs)

    def __len__(self) -> int:
        """Number of batches."""
        return self.total // self.batch_size + int(not self.drop_last and self.total % self.batch_size != 0)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Iterate over batches."""
        indices = torch.randperm(self.total) if self.shuffle else torch.arange(self.total)
        for i in range(0, self.total, self.batch_size):
            idx = indices[i : i + self.batch_size]
            batch = [trans(tensor[idx]) for tensor, trans in zip(self.tensors, self.transforms)]
            yield tuple(batch)
