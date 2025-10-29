import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Callable
from ulib.data.tensor_loader import TensorLoader
from ulib.utils.torch import extract_device

class DataExtractor:
    """
    Module to split a dataset by correctness, class label, or confidence using a PyTorch model.
    Assumes input tensors are first, and the target tensor is always last in the tuple.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the DataExtractor.

        Args:
            model (nn.Module): PyTorch model for predictions.
        """
        self.model = model.eval()
        self.device = extract_device(model)

    def _validate_loader(self, loader: TensorLoader) -> None:
        """Validate that loader has at least two tensors (inputs + target)."""
        if loader.num_tensors < 2:
            raise ValueError("Loader must contain at least one input tensor and one target tensor")

    def split_by_correctness(
        self,
        loader: TensorLoader,
        pred_func: Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor] | None = None,
    ) -> tuple[TensorLoader, TensorLoader]:
        """
        Split dataset into correct and incorrect predictions.

        Args:
            loader (TensorLoader): Dataset loader to split. Target tensor must be last.
            pred_func (Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor], optional):
                Custom prediction function taking model and all input tensors.
                Defaults to model(*inputs).argmax(dim=1) with all input tensors unpacked.

        Returns:
            tuple[TensorLoader, TensorLoader]: (correct_loader, incorrect_loader)
        """
        self._validate_loader(loader)

        # Initialize empty lists for each tensor
        cor_tensors = []
        inc_tensors = []
        for _ in range(loader.num_tensors):
            cor_tensors.append([])
            inc_tensors.append([])

        # Default pred_func unpacks all input tensors (all except the last)
        pred_func = pred_func or (lambda model, inputs: model(*inputs).argmax(dim=1))

        for batch in tqdm(loader, desc="Extracting by correctness"):
            # Prepare inputs and targets
            inputs = tuple(t.to(self.device) for t in batch[:-1])  # All except last
            targets = batch[-1].to(self.device)  # Last tensor is target
            preds = pred_func(self.model, inputs)

            mask = (preds == targets).cpu()
            batch = tuple(t.cpu() for t in batch)  # Move all tensors to CPU

            # Append to correct and incorrect lists
            for i in range(len(batch)):
                tensor = batch[i]
                cor_tensors[i].append(tensor[mask])
                inc_tensors[i].append(tensor[~mask])

        # handle case of empty lists (no correct or incorrect samples)
        for i, tlist in enumerate(cor_tensors):
            if len(tlist) == 0:
                tlist.append(torch.zeros((0, *loader.get_tensor(i).shape[1:])))

        for i, tlist in enumerate(inc_tensors):
            if len(tlist) == 0:
                tlist.append(torch.zeros((0, *loader.get_tensor(i).shape[1:])))

        cor_tensors = [torch.cat(tlist, dim=0) for tlist in cor_tensors]
        inc_tensors = [torch.cat(tlist, dim=0) for tlist in inc_tensors]

        return (
            TensorLoader(
                tuple(cor_tensors),
                batch_size=loader.batch_size,
                shuffle=loader.shuffle,
                drop_last=loader.drop_last,
                tensor_names=loader.tensor_names,
            ),
            TensorLoader(
                tuple(inc_tensors),
                batch_size=loader.batch_size,
                shuffle=loader.shuffle,
                drop_last=loader.drop_last,
                tensor_names=loader.tensor_names,
            ),
        )

    @torch.inference_mode()
    def extract_class(
        self,
        loader: TensorLoader,
        class_num: int | list[int],
    ) -> TensorLoader:
        """
        Extract samples of specific class(es).

        Args:
            loader (TensorLoader): Dataset loader to extract from. Target tensor must be last.
            class_num (int | list[int]): Class number(s) to extract.

        Returns:
            TensorLoader: Extracted dataset loader.
        """
        self._validate_loader(loader)

        class_nums = torch.tensor(class_num, dtype=loader.get_tensor(-1).dtype)
        idx = torch.isin(loader.get_tensor(-1), class_nums)

        # Extract tensors using explicit loop
        extracted_tensors = []
        for t in loader.tensors:
            extracted_tensors.append(t[idx])

        if not extracted_tensors[0].size(0):
            raise ValueError("No samples found for the specified class.")

        return TensorLoader(
            tuple(extracted_tensors),
            batch_size=loader.batch_size,
            shuffle=loader.shuffle,
            drop_last=loader.drop_last,
            tensor_names=loader.tensor_names,
        )

    def stratified_split(
        self,
        loader: TensorLoader,
        proportion: float,
        seed: int = 0,
    ) -> tuple[TensorLoader, TensorLoader]:
        """
        Split dataset into two parts maintaining class distribution.

        Args:
            loader (TensorLoader): Dataset loader to split. Target tensor must be last.
            proportion (float): Proportion for the first split.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple[TensorLoader, TensorLoader]: (split1, split2)
        """
        self._validate_loader(loader)
        if not (0.0 < proportion < 1.0):
            raise ValueError("proportion must be between 0 and 1")

        # Clone tensors
        tensors = []
        for t in loader.tensors:
            tensors.append(t.clone())
        targets = tensors[-1]  # Last tensor is target

        # Shuffle
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(targets), generator=rng)
        for i in range(len(tensors)):
            tensors[i] = tensors[i][perm]

        # Class distribution
        classes, counts = targets.unique(return_counts=True, sorted=False)
        class_counts = (proportion * counts.float()).floor().long()
        remainders = proportion * counts.float() - class_counts
        split_size = int(proportion * loader.total)
        num_leftover = split_size - class_counts.sum().item()

        if num_leftover > 0:
            _, idx_sorted = remainders.sort(descending=True)
            for idx in idx_sorted[:num_leftover]:
                class_counts[idx] += 1

        # Split tensors
        split_tensors = []
        other_tensors = []
        for _ in range(loader.num_tensors):
            split_tensors.append([])
            other_tensors.append([])

        for c, count in zip(classes, class_counts):
            idx = targets == c
            for i in range(len(tensors)):
                t = tensors[i]
                split_tensors[i].append(t[idx][:count])
                other_tensors[i].append(t[idx][count:])

        # Concatenate split tensors
        final_split_tensors = []
        final_other_tensors = []
        for i in range(len(split_tensors)):
            final_split_tensors.append(torch.cat(split_tensors[i], dim=0))
            final_other_tensors.append(torch.cat(other_tensors[i], dim=0))

        return (
            TensorLoader(
                tuple(final_split_tensors),
                batch_size=loader.batch_size,
                shuffle=loader.shuffle,
                drop_last=loader.drop_last,
                tensor_names=loader.tensor_names,
            ),
            TensorLoader(
                tuple(final_other_tensors),
                batch_size=loader.batch_size,
                shuffle=loader.shuffle,
                drop_last=loader.drop_last,
                tensor_names=loader.tensor_names,
            ),
        )

    @torch.inference_mode()
    def extract_confidence(
        self,
        loader: TensorLoader,
        threshold: float = 0.5,
        correct_only: bool = True,
    ) -> TensorLoader:
        """
        Extract low-confidence samples.

        Extraction strategy:
          - Compute the model's confidence of the ground-truth class (GT)
          - Compute the highest confidence for any class other than the GT
          - If (GT confidence - other confidence) <= threshold, consider the sample low-confidence

        Args:
            loader (TensorLoader): Dataset loader to extract from. Target tensor must be last.
            threshold (float): Confidence threshold.
            correct_only (bool): Extract only correct predictions.

        Returns:
            TensorLoader: Extracted dataset loader.
        """
        self._validate_loader(loader)

        # Initialize empty lists for each tensor
        extracted_tensors = []
        for _ in range(loader.num_tensors):
            extracted_tensors.append([])

        for batch in tqdm(loader, desc="Extracting by confidence"):
            # Prepare inputs and targets
            inputs = []
            for t in batch[:-1]:
                inputs.append(t.to(self.device))  # All except last
            targets = batch[-1].to(self.device)  # Last tensor is target
            logits = self.model(*inputs)  # Unpack all input tensors
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            if correct_only:
                cor_mask = preds == targets
                if not cor_mask.any():
                    continue
                new_batch = []
                for t in batch:
                    new_batch.append(t[cor_mask])
                batch = new_batch
                probs = probs[cor_mask]
                targets = targets[cor_mask]

            gt_conf = probs[torch.arange(len(targets)), targets].clone()
            probs[torch.arange(len(targets)), targets] = -1
            other_conf = probs.max(dim=1).values
            mask = (gt_conf - other_conf <= threshold).cpu()

            for i in range(len(batch)):
                t = batch[i]
                extracted_tensors[i].append(t[mask].cpu())

        if not extracted_tensors[0]:
            raise ValueError("No low-confidence samples found.")

        # Concatenate extracted tensors
        final_extracted_tensors = []
        for t_list in extracted_tensors:
            final_extracted_tensors.append(torch.cat(t_list, dim=0))

        return TensorLoader(
            tuple(final_extracted_tensors),
            batch_size=loader.batch_size,
            shuffle=loader.shuffle,
            drop_last=loader.drop_last,
            tensor_names=loader.tensor_names,
        )
