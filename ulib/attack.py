import torch
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import pathlib
import time
from typing import Iterable

from ulib.utils.torch import extract_device
from ulib.metric_logger import MetricLogger
from ulib.evaluator import MetricEvaluator, SimpleEvaluator, ExtendedEvaluator
from ulib.stop_criteria import StopCriteria
from ulib.pert_module import PertModule
from ulib.utils.logging import create_logger


logger = create_logger(__name__)


class UnivAttack(ABC):
    def __init__(
        self,
        pert_model: PertModule,
        targeted: bool = False,
        eval_freq: int | float = 1,
        metric_evaluator: MetricEvaluator | None = None,
        logging_enable: bool = False,
    ):
        """
        Base class for universal adversarial attacks. This class provides a framework for generating
        adversarial perturbations by iteratively training on the given dataset and evaluating its effectiveness.

        Subclasses must implement `process_batch` and may optionally override various hook methods.

        Args:
            pert_model (PertModule): A perturbation model used to generate adversarial examples.
            targeted (bool): Flag indicating if the attack is targeted or not.
            eval_freq (int | float): Frequency of evaluation during training.
                - if int, evaluates every `eval_freq` epochs.
                - if float, evaluates every `round(eval_freq * len(dl_train))` batches.
            metric_evaluator (MetricEvaluator, optional): Evaluator to compute the evaluation metrics.
            logging_enable (bool): If true, enables logging of metrics and hyperparameters.
        """
        if eval_freq <= 0:
            raise ValueError("Evaluation frequency must be greater than 0.")

        if metric_evaluator is None:
            metric_evaluator = SimpleEvaluator(pert_model, main_metric="misclassification_rate", verbose=True)

        self.pert_model = pert_model
        self.orig_model = pert_model.model
        self.targeted = targeted
        self.eval_freq = eval_freq
        self.metric_evaluator = metric_evaluator
        self.device = extract_device(pert_model)

        # Stats
        self.init_metric = -float("inf")
        self.best_metric = -float("inf")
        self.best_pert = pert_model.get_pert()

        # Logging
        self.metric_logger = MetricLogger(
            self.orig_model.__class__.__name__,
            self.__class__.__name__,
            f"L{self.pert_model.norm}",
            f"Eps_{self.pert_model.eps:.4f}",
            time.strftime("%Y-%m-%d_%H-%M-%S"),
            project="CLF-IML",
            root_dir="logs",
            disabled=not logging_enable,
        )

        self.metric_logger.report_hparams(
            "model",
            name=self.orig_model.__class__.__name__,
            device=self.device.type,
        )

        self.metric_logger.report_hparams(
            "attack",
            targeted=targeted,
            eval_freq=eval_freq,
            metric_name=metric_evaluator.name,
        )

        self.metric_logger.report_hparams(
            "metric_evaluator",
            metric_evaluator.get_hparams(),
        )

        self.metric_logger.report_hparams("part", self.pert_model.get_hparams())

    @property
    def judge_metric(self) -> str:
        """Returns the main metric name used for evaluation."""
        return self.metric_evaluator.default_metric

    def close(self):
        """Close all resources."""
        self.metric_logger.close()

    def __del__(self):
        self.close()

    def save_checkpoint(self, file_name: str = "best_pert.pt"):
        if self.metric_logger.log_dir is None:
            logger.warning("Log dir is None, cannot save checkpoint.")
            return

        torch.save(self.best_pert, pathlib.Path(self.metric_logger.log_dir) / file_name)

    def to_device(self, *data: torch.Tensor, device: str | torch.device | None = None) -> tuple[torch.Tensor, ...]:
        """
        Moves the given tensors to device.

        Args:
            data (torch.Tensor): Input tensors to move to the device.
            device (str | torch.device | None): Device to move the tensors to. If None, uses `self.device`.

        Returns:
            tuple[torch.Tensor, ...]: Tensors moved to the device.
        """
        device = self.device if device is None else device
        return tuple(arg.to(device) for arg in data)

    @torch.inference_mode()
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        """
        Evaluates the current perturbation model on the given dataset and returns the metric value.

        Args:
            dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation data loader.

        Returns:
            dict[str, float]: A dictionary mapping metric names to their computed values.
        """
        metrics = self.metric_evaluator.evaluate(dl_eval)
        value = metrics[self.judge_metric]
        if value > self.best_metric:
            self.best_metric = value
            self.best_pert = self.pert_model.get_pert(clone=True)
        return metrics

    def fit(
        self,
        dl_train: Iterable[tuple[torch.Tensor, ...]],
        dl_eval: Iterable[tuple[torch.Tensor, ...]] | None = None,
        stop_criteria: StopCriteria | None = None,
    ) -> torch.Tensor:
        """
        Generates the adversarial perturbation by iteratively training on the given dataset and evaluating
        its effectiveness.

        Args:
            dl_train (Iterable[tuple[torch.Tensor, ...]]): Training data loader.
            dl_eval (Iterable[tuple[torch.Tensor, ...]], optional): Evaluation data loader. If None, uses `dl_train`.
            stop_criteria (StopCriteria, optional): Stopping criteria for training.

        Returns:
            torch.Tensor: Best adversarial perturbation found.
        """
        if dl_eval is None:
            dl_eval = dl_train

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        stop_criteria.reset()

        # Init Logging
        self.metric_logger.report_hparams("stop_criteria", stop_criteria.get_hparams())

        self.metric_logger.set_tags(
            model=self.orig_model.__class__.__name__,
            attack=self.__class__.__name__,
            norm=self.pert_model.norm,
            eps=f"{self.pert_model.eps:4f}",
        )

        # Init Stats
        metrics = self.metric_evaluator.evaluate(dl_eval)
        self.init_metric = metrics[self.judge_metric]
        self.best_metric = self.init_metric
        self.best_pert = self.pert_model.get_pert()

        self.metric_logger.report_scalars({f"metric/{k}": v for k, v in metrics.items()}, step=-1)
        self.metric_logger.report_scalar(f"metric/{self.judge_metric}/best", self.best_metric, step=-1)

        loss = None
        main_metric = self.init_metric
        should_stop = False
        step = 0

        self.pert_model.train()

        epoch_pbar = tqdm(range(stop_criteria.max_epochs), desc="Epochs")
        for epoch_num in epoch_pbar:
            if should_stop:
                break

            batch_pbar = tqdm(dl_train, desc="Batches", leave=False)
            for batch_num, data in enumerate(batch_pbar):
                if should_stop:
                    break

                data = self.to_device(*data)
                with torch.device(self.device):
                    loss = self.process_batch(data, batch_num=batch_num, epoch_num=epoch_num, step_num=step)
                stop_criteria.update(epoch_num, None)
                self.metric_logger.report_scalar("loss/current", loss, step)

                if should_stop or (step > 0 and step % round(self.eval_freq * len(dl_train)) == 0):
                    metrics = self.evaluate(dl_eval)
                    main_metric = metrics[self.judge_metric]
                    stop_criteria.update(epoch_num, main_metric)
                    should_stop = stop_criteria.should_stop()

                    self.save_checkpoint()
                    self.metric_logger.report_scalars({f"metric/{k}": v for k, v in metrics.items()}, step)
                    self.metric_logger.report_scalar(f"metric/{self.judge_metric}/best", self.best_metric, step)
                    self.metric_logger.report_image("pert/current", self.pert_model.to_image(), step)

                batch_pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}" if loss is not None else "None",
                        f"{self.judge_metric}_init": f"{self.init_metric:.4f}",
                        f"{self.judge_metric}_best": f"{self.best_metric:.4f}",
                        f"{self.judge_metric}_curr": f"{main_metric:.4f}",
                    }
                )

                step += 1

            batch_pbar.close()

            epoch_pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}" if loss is not None else "None",
                    f"{self.judge_metric}_init": f"{self.init_metric:.4f}",
                    f"{self.judge_metric}_best": f"{self.best_metric:.4f}",
                    f"{self.judge_metric}_curr": f"{main_metric:.4f}",
                }
            )

        epoch_pbar.close()

        # Update to best pert
        self.pert_model.set_pert(self.best_pert)

        # Final evaluation
        extra_evaluator = ExtendedEvaluator(self.pert_model, verbose=True)
        metrics = extra_evaluator.evaluate(dl_eval)
        metrics = {k: round(v, 4) for k, v in metrics.items()}
        self.metric_logger.report_globals(metrics)
        self.metric_logger.report_image("pert/best", self.pert_model.to_image(), step)

        self.close()
        return self.pert_model.get_pert()

    @abstractmethod
    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:
        """
        Runs a single training step on the given batch of data.

        Args:
            data (tuple[torch.Tensor, ...]): Batch data, already moved to the device.
            batch_num (int): Current batch number.
            epoch_num (int): Current epoch number.
            step_num (int): Current global step number.

        Returns:
            float: Computed loss value for the batch, or None if invalid.
        """
        raise NotImplementedError()


class OptimAttack(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        sched_on_batch: bool = False,
        grad_scaler: torch.GradScaler | None = None,
        autocast: torch.autocast | None = None,
        **kwargs,
    ):
        """
        Base class for universal adversarial attacks that uses an optimizer.
        This class provides a framework for generating adversarial perturbations by training
        on a given dataset and evaluating its effectiveness.

        Args:
            pert_model (PertModule): Perturbation model used to generate adversarial examples.
            optimizer (torch.optim.Optimizer): Optimizer for training the perturbation model.
            criterion (torch.nn.Module): Loss function for computing the training loss.
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler.
            sched_on_batch (bool): If True, steps the scheduler on every batch, otherwise on every epoch.
            grad_scaler (torch.GradScaler, optional): Gradient scaler for mixed precision.
            autocast (torch.autocast, optional): Autocast for mixed precision training.
            targeted (bool): Flag indicating if the attack is targeted or not.
            eval_freq (int | float): Frequency of evaluation during training.
                - if int, evaluates every `eval_freq` epochs.
                - if float, evaluates every `round(eval_freq * len(dl_train))` batches.
            metric_evaluator (MetricEvaluator, optional): Evaluator to compute the evaluation metrics.
            logging_enable (bool): If true, enables logging of metrics and hyperparameters.
            **kwargs: Additional arguments passed to the `UnivAttack` constructor.
        """
        super().__init__(
            pert_model=pert_model,
            **kwargs,
        )

        if autocast is not None and grad_scaler is None:
            raise ValueError("Grad scaler must be provided when using autocast.")

        if autocast is None:
            enabled = grad_scaler is not None and grad_scaler.is_enabled()
            autocast = torch.autocast(device_type=self.device.type, enabled=enabled)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.sched_on_batch = sched_on_batch
        self.grad_scaler = grad_scaler
        self.autocast = autocast

        self.metric_logger.report_hparams(
            "optim",
            self.optimizer.param_groups[0],
            name=self.optimizer.__class__.__name__,
        )
        self.metric_logger.report_hparams(
            "criterion",
            self.criterion.__dict__,
            name=self.criterion.__class__.__name__,
        )
        if self.scheduler is not None:
            self.metric_logger.report_hparams(
                "scheduler",
                self.scheduler.state_dict(),
                name=self.scheduler.__class__.__name__,
            )
        if self.grad_scaler is not None:
            self.metric_logger.report_hparams("grad_scaler", self.grad_scaler.state_dict())
        if self.autocast is not None:
            self.metric_logger.report_hparams("autocast", self.autocast.__dict__)

    def autocast_context(self, enabled: bool = True) -> torch.autocast:
        """
        Returns an autocast context manager with the given enabled state.
        - If `enabled=True` then autocast will be enabled only if `self.autocast.is_enabled()` is True.
        - If `enabled=False` then autocast will be disabled.

        Args:
            enabled (bool): Whether to enable autocast.

        Returns:
            torch.autocast: Autocast context manager.
        """
        if enabled:
            return self.autocast
        else:
            return torch.autocast(device_type=self.autocast.device, enabled=False)

    @abstractmethod
    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor | None:
        """
        Computes the loss on the given batch of data.
        The computed loss must be a scalar tensor that allows gradients to be backpropagated.

        Args:
            data (tuple[torch.Tensor, ...]): Batch data, already moved to the device.
            batch_num (int): Current batch number.
            epoch_num (int): Current epoch number.
            step_num (int): Current global step number.

        Returns:
            torch.Tensor: Loss tensor, or None if invalid.
        """
        raise NotImplementedError()

    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:
        self.optimizer.zero_grad()

        with self.autocast_context():
            loss = self.compute_loss(data, batch_num, epoch_num, step_num)

        if loss is not None:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        # Update learning rate scheduler if enabled.
        if self.scheduler is not None and (self.sched_on_batch or batch_num == 0):
            self.metric_logger.report_scalar("lr", self.scheduler.get_last_lr()[0], step_num)
            self.scheduler.step()

        return None if loss is None else loss.item()

    @torch.inference_mode()
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        with self.autocast_context():
            return super().evaluate(dl_eval)
