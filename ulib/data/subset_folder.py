import os
import random
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Callable, Optional, Union
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, find_classes, IMG_EXTENSIONS


def make_partial_dataset(
    directory: str | Path,
    total_size: int,
    class_to_idx: dict[str, int] | None = None,
    extensions: str | tuple[str, ...] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
    allow_empty: bool = False,
) -> list[tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    if extensions is None and is_valid_file is None:
        raise ValueError("Both extensions and is_valid_file cannot both be None at the same time")

    if extensions is not None and is_valid_file is not None:
        raise ValueError("Both extensions and is_valid_file cannot both be not None at the same time")

    if is_valid_file is None:
        is_valid_file = lambda x: has_file_allowed_extension(x, extensions)  # type: ignore

    num_classes = len(class_to_idx)
    base_size = total_size // num_classes
    leftover_size = total_size % num_classes

    pbar = tqdm(total=total_size, desc="Creating Dataset", leave=False)

    # randomly pick leftover_count classes to get 1 extra sample
    leftover_classes = set(random.sample(range(num_classes), leftover_size))

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        class_size = base_size + (1 if class_index in leftover_classes else 0)
        collected = 0

        if class_size == 0:
            continue

        class_dir = os.path.join(directory, target_class)

        if not os.path.isdir(class_dir):
            continue

        with os.scandir(class_dir) as dir_it:
            for entry in dir_it:
                if not entry.is_file():
                    continue
                if not is_valid_file(entry.path):
                    continue

                if collected >= class_size:
                    # stop scanning this folder
                    break

                instances.append((entry.path, class_index))
                available_classes.add(target_class)
                collected += 1
                pbar.update(1)

    pbar.close()

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        if extensions is None:
            extensions = ".*"

        msg = f"""Some classes are empty while `allow_empty=False`.
        Following classes are empty: {", ".join(sorted(empty_classes))}.
        Supported extensions are: {extensions if isinstance(extensions, str) else ", ".join(extensions)}"""
        raise FileNotFoundError(msg)

    return instances


class SubsetFolder(DatasetFolder):
    """
    A DatasetFolder-like dataset that only loads a total of `size` samples,
    distributing them roughly evenly across classes, and only enumerating
    enough files from each class directory to fulfill the needed count.

    Args:
        root (str or Path): Root directory path.
        size (int): Total number of samples to load across all classes.
        loader (callable): A function to load a sample given its path.
        extensions (tuple, optional): A list of allowed extensions.
            If `is_valid_file` is also provided, this argument is ignored.
        transform (callable, optional): Transform function for the samples.
        target_transform (callable, optional): Transform function for the targets.
        is_valid_file (callable, optional): Used to check if a file is valid.
        allow_empty (bool, optional): If True, empty folders are allowed (default=False).

    Note: This class overrides the `make_dataset` method to return an empty list.
    """

    def __init__(
        self,
        root: Union[str, Path],
        size: int,
        loader: Callable[[str], Any],
        extensions: Optional[tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root=root,
            loader=loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        partial_samples = make_partial_dataset(
            directory=self.root,
            total_size=size,
            class_to_idx=self.class_to_idx,
            extensions=self.extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.samples = partial_samples
        self.targets = [s[1] for s in partial_samples]

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: dict[str, int],
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> list[tuple[str, int]]:
        # We do not implement partial logic here, because we cannot read
        # self.partial_size from a static method directly.
        # Instead, we call make_partial_dataset() after the base class is initialized.
        return []


class PartialImageFolder(SubsetFolder):
    """
    Similar to torchvision.datasets.ImageFolder, but loads only `size` total images,
    balanced across classes. Minimizes directory scanning by stopping early in each class folder.

    Args:
        root (str or Path): Root directory path.
        size (int): Total number of samples to load across all classes.
        transform (callable, optional): Transform for images.
        target_transform (callable, optional): Transform for targets.
        loader (callable, optional): Custom image loader (defaults to torchvision's default_loader).
        is_valid_file (callable, optional): Function to check if a file is valid.
        allow_empty (bool, optional): If True, empty folders are allowed.

    Example:
        folder = PartialImageFolder(root="mydata", size=100)
        img, label = folder[0]
    """

    def __init__(
        self,
        root: Union[str, Path],
        size: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        extensions = None if is_valid_file is not None else IMG_EXTENSIONS

        super().__init__(
            root=root,
            size=size,
            loader=loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.imgs = self.samples
