from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import torch
from typing import Iterable
from ulib.utils.torch import extract_device
from ulib.pert_module import PertModule


class MetricEvaluator(ABC):
    def __init__(self, model: torch.nn.Module, main_metric: str | None = None, verbose: bool = True):
        """
        A base class for evaluators that processes batches of data and computes evaluation metrics.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            main_metric (str | None): The primary metric to focus on during evaluation.
            verbose (bool): If True, enables verbose output during evaluation.
        """
        self.model = model
        self.main_metric = main_metric
        self.verbose = verbose
        self.device = extract_device(self.model)

        if self.main_metric is None:
            self.main_metric = self.default_metric

    def get_hparams(self) -> dict:
        return {
            "name": self.name,
            "main_metric": self.main_metric,
            "verbose": self.verbose,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the evaluator.

        Returns:
            str: Name of the evaluator.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    @abstractmethod
    def metric_names(self) -> list[str]:
        """
        Returns the list of metric names produced by the evaluator.

        Returns:
            list[str]: List of metric names.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def default_metric(self) -> str:
        """
        Returns the default metric name for this evaluator.

        Returns:
            str: Default metric name.
        """
        return self.main_metric or self.metric_names[0]

    @abstractmethod
    def eval_batch(self, data: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """
        Evaluates a single batch of data and computes the metrics for every element in the batch.

        Args:
            data (tuple[torch.Tensor, ...]): A batch of data.

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping metric names to their computed values for the batch.
                Tensors of shape `(batch_size,)`.

        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @torch.inference_mode()
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        """ """
        mode = self.model.training
        self.model.eval()

        sums: dict[str, float] = {k: 0.0 for k in self.metric_names}
        counts: dict[str, float] = {k: 0.0 for k in self.metric_names}

        for batch_data in tqdm(dl_eval, desc=f"Evaluating {self.name}", disable=not self.verbose, leave=False):
            batch_data = tuple(t.to(self.device) for t in batch_data)
            batch_metric = self.eval_batch(batch_data)
            for metric_name, values in batch_metric.items():
                sums[metric_name] += torch.nansum(values.float()).item()
                counts[metric_name] += torch.sum(~torch.isnan(values)).item()

        self.model.train(mode)
        return {k: sums[k] / counts[k] if counts[k] > 0 else float("nan") for k in self.metric_names}


class SimpleEvaluator(MetricEvaluator):
    def __init__(
        self,
        model: torch.nn.Module,
        main_metric: str | None = None,
        verbose: bool = True,
    ):
        super().__init__(model, main_metric, verbose)

    @property
    def name(self) -> str:
        return "Simple Metrics"

    @property
    def metric_names(self) -> list[str]:
        return ["ACC", "MR"]

    @property
    def default_metric(self) -> str:
        return self.main_metric or "MR"

    def eval_batch(self, data: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        x, y = data
        preds = self.model(x).argmax(dim=1)

        correct = preds == y
        incorrect = ~correct

        return {
            "ACC": correct.to(torch.float32),  # accuracy
            "MR": incorrect.to(torch.float32),  # misclassification rate
        }


class ExtendedEvaluator(MetricEvaluator):
    def __init__(self, model: PertModule, main_metric: str | None = None, verbose: bool = True):
        assert isinstance(model, PertModule), "model must be a PertModule."
        super().__init__(model, main_metric, verbose)

    @property
    def name(self) -> str:
        return "Extended Metrics"

    @property
    def metric_names(self) -> list[str]:
        return [
            "FR",  # fooling rate
            "ASR",  # attack success rate
            "C-ACC",  # clean accuracy
            "R-ACC",  # robust accuracy
            "C-MR",  # clean misclassification rate
            "R-MR",  # robust misclassification rate
        ]

    @property
    def default_metric(self) -> str:
        return "ASR"

    def eval_batch(self, data: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        x, y = data
        pert_module: PertModule = self.model  # type: ignore
        pred_cln = pert_module.model(x).argmax(dim=1)
        pred_adv = pert_module(x).argmax(dim=1)

        fooled = pred_cln != pred_adv
        clean_correct = pred_cln == y
        robust_correct = pred_adv == y
        clean_incorrect = ~clean_correct
        robust_incorrect = ~robust_correct

        return {
            "FR": fooled.to(torch.float32),
            "C-ACC": clean_correct.to(torch.float32),
            "R-ACC": robust_correct.to(torch.float32),
            "C-MR": clean_incorrect.to(torch.float32),
            "R-MR": robust_incorrect.to(torch.float32),
        }

    @torch.inference_mode()
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        metrics = super().evaluate(dl_eval)

        cac = metrics["C-ACC"]
        rac = metrics["R-ACC"]
        metrics["ASR"] = (cac - rac) / cac if cac > 0 else float("nan")
        return metrics
