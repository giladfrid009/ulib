from typing import Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from tqdm.auto import tqdm

from ulib.pert_module import PertModule
from ulib.utils.torch import extract_device
from ulib.utils.logging import create_logger


logger = create_logger(__name__)


def display_pert(pert_module: PertModule) -> AxesImage:
    """
    Display the perturbation as an image.

    Args:
        pert_module (PertModule): Perturbation to display.

    Returns:
        AxesImage: Plot of the perturbation.
    """

    p = pert_module.to_image().numpy()
    p = np.transpose(p, (1, 2, 0))
    ax = plt.imshow(p)
    plt.axis("off")
    return ax


@torch.inference_mode()
def accuracy(
    model: torch.nn.Module,
    dl_eval: Iterable[tuple[torch.Tensor, ...]],
    silent: bool = False,
) -> float:
    """
    Calculate the accuracy of a model on an evaluation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation loader.
        silent (bool): Whether to use tqdm to display progress.

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    mode = model.training
    model.eval()
    device = extract_device(model)

    total = 0
    correct = torch.zeros(1, device=device, dtype=torch.int64)
    for x, y in tqdm(dl_eval, desc="Evaluating", leave=False, disable=silent):
        x, y = x.to(device), y.to(device)
        y_pred = model(x).argmax(dim=1)
        correct += (y_pred == y).sum()
        total += len(y)

    model.train(mode)
    return correct.item() / total


@torch.inference_mode()
def misclassification_rate(
    model: torch.nn.Module,
    dl_eval: Iterable[tuple[torch.Tensor, ...]],
    silent: bool = False,
) -> float:
    """
    Calculate the misclassification rate of a model on an evaluation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation loader.
        silent (bool): Whether to use tqdm to display progress.
    """
    return 1.0 - accuracy(model, dl_eval, silent)


@torch.inference_mode()
def fooling_rate(
    pert_module: PertModule,
    dl_eval: Iterable[tuple[torch.Tensor, ...]],
    silent: bool = False,
) -> float:
    """
    Calculate the fooling rate of a perturbation module on an evaluation dataset.

    Args:
        pert_module (PertModule): The perturbation module to evaluate.
        dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation loader.
        silent (bool): Whether to use tqdm to display progress.

    Returns:
        float: The fooling rate of the perturbation module on the evaluation dataset.
    """
    mode = pert_module.training
    pert_module.eval()
    device = extract_device(pert_module)

    total = 0
    mismatches = torch.zeros(1, device=device, dtype=torch.int64)
    for x, y in tqdm(dl_eval, desc="Evaluating", leave=False, disable=silent):
        x, y = x.to(device), y.to(device)
        pred_cln = pert_module.model(x).argmax(dim=1)
        pred_adv = pert_module(x).argmax(dim=1)
        mismatches += (pred_cln != pred_adv).sum()
        total += len(y)

    pert_module.train(mode)
    return mismatches.item() / total


@torch.inference_mode()
def attack_success_ratio(
    pert_module: PertModule,
    dl_eval: Iterable[tuple[torch.Tensor, ...]],
    silent: bool = False,
) -> float:
    """
    Calculate the attack success ratio of a perturbation module on an evaluation dataset.

    Args:
        pert_module (PertModule): The perturbation module to evaluate.
        dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation loader.
        silent (bool): Whether to use tqdm to display progress.

    Returns:
        float: The attack success ratio of the perturbation module on the evaluation dataset.
    """
    mode = pert_module.training
    pert_module.eval()
    device = extract_device(pert_module)

    total = 0
    corr_cln = torch.zeros(1, device=device, dtype=torch.int64)
    corr_adv = torch.zeros(1, device=device, dtype=torch.int64)
    for x, y in tqdm(dl_eval, desc="Evaluating", leave=False, disable=silent):
        x, y = x.to(device), y.to(device)
        pred_cln = pert_module.model(x).argmax(dim=1)
        pred_adv = pert_module(x).argmax(dim=1)
        corr_cln += (pred_cln == y).sum()
        corr_adv += (pred_adv == y).sum()
        total += len(y)

    pert_module.train(mode)
    clean_acc = corr_cln.item() / total
    robust_acc = corr_adv.item() / total
    return (clean_acc - robust_acc) / clean_acc


@torch.inference_mode()
def full_analysis(
    pert_module: PertModule,
    dl_eval: Iterable[tuple[torch.Tensor, ...]],
    silent: bool = False,
) -> dict[str, float]:
    """
    Perform a full analysis of a perturbation module on an evaluation dataset.

    Args:
        pert_module (PertModule): The perturbation module to evaluate.
        dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation loader.
        silent (bool): Whether to use tqdm to display progress.

    Returns:
        dict[str, float]: Results of the analysis, where keys are metric names and values are the corresponding metric values.
    """
    cac = accuracy(pert_module.model, dl_eval, silent)
    rac = accuracy(pert_module, dl_eval, silent)
    asr = (cac - rac) / cac
    fr = fooling_rate(pert_module, dl_eval, silent)

    results = {
        "clean_accuracy": round(cac, 4),
        "robust_accuracy": round(rac, 4),
        "attack_success_ratio": round(asr, 4),
        "fooling_rate": round(fr, 4),
    }

    for metric, value in results.items():
        logger.info(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")

    display_pert(pert_module)
    plt.show()

    return results
