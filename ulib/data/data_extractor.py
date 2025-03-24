# inspired by https://arxiv.org/pdf/1901.04684
# to extract low-confidence samples from a dataset
# and construct a pertubations for them only

import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Callable
from ulib import utils
from ulib.data.tensor_loader import TensorLoader


class DataExtractor:
    """
    Module to split a dataset by correctness, class label, or confidence, using a PyTorch model.
    Provides functionalities to filter or sample datasets for adversarial training or evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:

        self.model = model.eval()
        self.device = utils.extract_device(model)

    def split_by_correctness(
        self,
        loader: TensorLoader,
        pred_func: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[TensorLoader, TensorLoader]:
        """
        Split a dataset into correct and incorrectly predicted samples.

        Args:
            loader (TensorLoader): Dataset loader to split.
            pred_func (Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor], optional): Prediction function to use.
                Defaults to lambda model, x, y: model(x).argmax(dim=1).

        Returns:
            tuple[TensorLoader, TensorLoader]: (correct_x, correct_y), (incorrect_x, incorrect_y)
        """
        cor_x = []
        cor_y = []
        inc_x = []
        inc_y = []

        if pred_func is None:
            pred_func = lambda model, x, y: model(x).argmax(dim=1)

        for x_batch, y_batch in tqdm(loader, desc="Extracting"):
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            preds = pred_func(self.model, x_batch, y_batch)

            mask = (preds == y_batch).cpu()
            x_batch, y_batch = x_batch.cpu(), y_batch.cpu()

            cor_x.append(x_batch[mask])
            cor_y.append(y_batch[mask])
            inc_x.append(x_batch[~mask])
            inc_y.append(y_batch[~mask])

        if len(cor_x) == 0:
            cor_x = torch.zeros((0, *loader.data.shape[1:]))
            cor_y = torch.zeros((0,), device=self.device)
        else:
            cor_x = torch.stack(cor_x, dim=0)
            cor_y = torch.stack(cor_y, dim=0)

        if len(inc_x) == 0:
            inc_x = torch.zeros((0, *loader.data.shape[1:]))
            inc_y = torch.zeros((0,), device=self.device)
        else:
            inc_x = torch.stack(inc_x, dim=0)
            inc_y = torch.stack(inc_y, dim=0)

        dl_cor = TensorLoader(cor_x, cor_y, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)
        dl_inc = TensorLoader(inc_x, inc_y, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)

        return dl_cor, dl_inc

    @torch.inference_mode()
    def extract_class(self, loader: TensorLoader, class_num: int | list[int]) -> TensorLoader:
        """
        Extract samples of a specific class(es) from a dataset.

        Args:
            loader (TensorLoader): Dataset loader to extract from
            class_num (int | list[int]): Class number(s) to extract

        Returns:
            TensorLoader: Extracted dataset loader
        """
        class_nums = torch.tensor(class_num, dtype=loader.targets.dtype)

        idx = torch.isin(loader.targets, class_nums)
        extr_x = loader.data[idx]
        extr_y = loader.targets[idx]

        if len(extr_x) == 0:
            raise ValueError("No samples found for the specified class.")

        return TensorLoader(extr_x, extr_y, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)

    def stratified_split(self, loader: TensorLoader, proportion: float, seed: int = 0) -> tuple[TensorLoader, TensorLoader]:
        """
        Split a dataset into two parts while maintaining the class distribution.

        Args:
            loader (TensorLoader): Dataset loader to split.
            proportion (float): Proportion of the first split.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple[TensorLoader, TensorLoader]: (split1, split2)
        """

        if not (0.0 < proportion < 1.0):
            raise ValueError("proportion must be between 0 and 1")

        data = loader.data.clone()
        targets = loader.targets.clone()

        # Shuffle the dataset
        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(len(targets), generator=rng)
        data = data[perm]
        targets = targets[perm]

        # Count the number of samples in each class
        classes, total_class_counts = targets.unique(return_counts=True, sorted=False)
        classes: torch.Tensor
        total_class_counts: torch.Tensor

        # Compute the number of samples to place in the first split
        class_counts_float = proportion * total_class_counts.float()
        class_counts = class_counts_float.floor().long()
        remainders = class_counts_float - class_counts

        # Distribute leftover elements to classes with largest fractional remainder
        split_size = int(proportion * loader.total)
        num_leftover = split_size - class_counts.sum().item()
            
        if num_leftover > 0:
            _, idx_sorted = remainders.sort(descending=True)
            idx_top = idx_sorted[:num_leftover]
            class_counts[idx_top] += 1

        # Split the dataset
        split_data = []
        split_targets = []
        other_data = []
        other_targets = []

        for c in tqdm(classes, desc="Splitting"):
            idx = targets == c
            idx_split = idx[: class_counts[c]]
            idx_other = idx[class_counts[c] :]

            split_data.append(data[idx_split])
            split_targets.append(targets[idx_split])
            other_data.append(data[idx_other])
            other_targets.append(targets[idx_other])

        split_data = torch.cat(split_data, dim=0)
        split_targets = torch.cat(split_targets, dim=0)
        other_data = torch.cat(other_data, dim=0)
        other_targets = torch.cat(other_targets, dim=0)

        dl1 = TensorLoader(split_data, split_targets, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)
        dl2 = TensorLoader(other_data, other_targets, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)

        return dl1, dl2

    @torch.inference_mode()
    def extract_confidence(
        self,
        loader: TensorLoader,
        threshold: float = 0.5,
        correct_only: bool = True,
    ) -> TensorLoader:
        """
        Extract low-confidence samples from a dataset.

        Extraction strategy:
          - Compute the model's confidence of the ground-truth class (GT)
          - Compute the highest confidence for any class other than the GT
          - If (GT confidence - other confidence) <= threshold,
             consider the sample low-confidence

        Args:
            loader (TensorLoader): Dataset loader to extract from.
            threshold (float, optional): Confidence threshold.
            correct_only (bool, optional): Extract only correct predictions.

        Returns:
            TensorLoader: Extracted dataset loader
        """
        extr_x = []
        extr_y = []

        for x_batch, y_batch in tqdm(loader, desc="Extracting"):
            # Move data to device
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # Only use correct predictions
            if correct_only:
                cor_mask = preds == y_batch
                if not cor_mask.any():
                    continue

                x_batch, y_batch = x_batch[cor_mask], y_batch[cor_mask]
                probs = probs[cor_mask]

            # Compute the confidence of the ground-truth class
            idx = torch.arange(len(y_batch), device=self.device)
            gt_conf = probs[idx, y_batch].clone()

            # Temporarily set GT confidence to -1 to find the highest non-GT confidence
            probs[idx, y_batch] = -1
            other_conf = probs.max(dim=1).values

            # Low confidence mask
            mask = gt_conf - other_conf <= threshold

            extr_x.append(x_batch[mask].cpu())
            extr_y.append(y_batch[mask].cpu())

        if len(extr_x) == 0:
            raise ValueError("No low-confidence samples found.")

        # Convert lists to tensors
        extr_x = torch.stack(extr_x, dim=0)
        extr_y = torch.stack(extr_y, dim=0)

        return TensorLoader(extr_x, extr_y, batch_size=loader.batch_size, shuffle=loader.shuffle, drop_last=loader.drop_last)
