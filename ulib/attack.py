import torch
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import pathlib
import time
from typing import Iterable, Callable

from ulib.utils.torch import extract_device
from ulib.metric_logger import MetricLogger
from ulib import eval
from ulib.pert_module import PertModule
from ulib.utils.logging import create_logger


logger = create_logger(__name__)


class StopCriteria:
    def __init__(
        self,
        max_epochs: int = 100,
        max_evals: int | None = None,
        max_time: float | None = None,
        target_value: float | None = None,
        patience: int | None = None,
        patience_delta: float = 1e-4,
    ):
        """
        Container for various stopping criteria.

        Args:
            max_epochs: Maximum number of epochs.
            max_evals: Maximum number of evaluation steps.
            max_time: Maximum training time in seconds.
            target_value: Target value to stop training when reached. This value should be maximized.
            patience: Number of evaluation steps without sufficient improvement.
            patience_delta: Minimum improvement delta to reset patience.

        Raises:
            ValueError: If any of the arguments are invalid.
        """
        if max_epochs <= 0:
            raise ValueError("Max epochs must be greater than 0.")
        if max_evals is not None and max_evals <= 0:
            raise ValueError("Max evals must be greater than 0.")
        if max_time is not None and max_time <= 0:
            raise ValueError("Max time must be greater than 0.")
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be greater than 0.")
        if patience_delta < 0:
            raise ValueError("Patience delta must be greater than or equal to 0.")

        self.max_epochs = max_epochs
        self.max_evals = max_evals
        self.max_time = max_time
        self.target_value = target_value
        self.patience = patience
        self.patience_delta = patience_delta

        # Internal state
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0
        self.reset()

    def get_hparams(self) -> dict:
        return {
            "max_epochs": self.max_epochs,
            "max_evals": self.max_evals,
            "max_time": self.max_time,
            "target_value": self.target_value,
            "patience": self.patience,
            "patience_delta": self.patience_delta,
        }

    def reset(self) -> None:
        """Reset internal state."""
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0

    def update(self, epoch: int, value: float) -> None:
        """Update internal state with new metrics."""
        self._epoch = epoch
        self._total_evals += 1

        if (value - self._best_value) >= self.patience_delta:
            self._best_value = value
            self._patience_counter = 0
        else:
            self._patience_counter += 1

    def should_stop(self) -> bool:
        """Check if any stopping condition is met."""
        if self.target_value is not None and self._best_value >= self.target_value:
            print(f"Stopping: Target value reached :: ({self.target_value})")
            return True

        if self._epoch >= self.max_epochs:
            print(f"Stopping: Max epochs reached :: ({self.max_epochs})")
            return True

        if self.max_evals is not None and self._total_evals >= self.max_evals:
            print(f"Stopping: Max evals reached :: ({self.max_evals})")
            return True

        if self.patience is not None and self._patience_counter >= self.patience:
            print(f"Stopping: Patience exceeded :: ({self.patience})")
            return True

        if self.max_time is not None and (time.time() - self._start_time) > self.max_time:
            print(f"Stopping: Max time reached :: ({self.max_time} sec)")
            return True

        return False


class UnivAttack(ABC):
    def __init__(
        self,
        pert_model: PertModule,
        targeted: bool = False,
        eval_freq: int = 1,
        eval_on_batch: bool = False,
        metric_name: str = "metric",
        metric_func: Callable[[PertModule, Iterable[tuple[torch.Tensor, ...]]], float] | None = None,
        log_dir: str | None = None,
    ):
        """
        Base class for universal adversarial attacks. This class provides a framework for generating
        adversarial perturbations by iteratively training on the given dataset and evaluating its effectiveness.

        Subclasses must implement `process_batch` and may optionally override various hook methods.

        Args:
            pert_model (PertModule): A perturbation model used to generate adversarial examples.
            targeted (bool): Flag indicating if the attack is targeted or not.
            eval_freq (int): Every how many steps to evaluate the model performance.
            eval_on_batch (bool): If true, each batch is counted as step for evaluation, otherwise each epoch.
            metric_name (str): Name of the evaluation metric.
            metric_func (Callable[[PertModule, Iterable[tuple[torch.Tensor, ...]]], float], optional): Function to compute the evaluation metric.
            log_dir (str, optional): Directory for storing logs. If None, logging is disabled.
        """
        if eval_freq <= 0:
            raise ValueError("Evaluation frequency must be greater than 0.")

        if metric_func is None:
            metric_func = eval.misclassification_rate

        self.pert_model = pert_model
        self.orig_model = pert_model.model
        self.targeted = targeted
        self.eval_freq = eval_freq
        self.eval_on_batch = eval_on_batch
        self.metric_name = metric_name
        self.metric_func = metric_func
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
            project="CLF-IML",
            root_dir="logs",
            disabled=log_dir is None,  # TODO: fix log_dir arg afterwards
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
            eval_on_batch=eval_on_batch,
            metric_name=metric_name,
            metric_func=metric_func.__name__,
        )

        self.metric_logger.report_hparams("part", self.pert_model.get_hparams())

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
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> float:
        """
        Evaluates the current perturbation model on the given dataset and returns the metric value.

        Args:
            dl_eval (Iterable[tuple[torch.Tensor, ...]]): The evaluation data loader.

        Returns:
            float: The metric value of the perturbation model on the evaluation dataset.
        """
        value = self.metric_func(self.pert_model, dl_eval)
        if value > self.best_metric:
            self.best_metric = value
            self.best_pert = self.pert_model.get_pert(clone=True)
        return value

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
        self.init_metric = self.metric_func(self.pert_model, dl_eval)
        self.best_metric = self.init_metric
        self.best_pert = self.pert_model.get_pert()

        self.metric_logger.report_scalar(f"{self.metric_name}/current", self.init_metric, step=-1)
        self.metric_logger.report_scalar(f"{self.metric_name}/best", self.best_metric, step=-1)

        loss = None
        current_metric = self.init_metric
        should_stop = False
        global_step = 0

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
                loss = self.process_batch(data, batch_num=batch_num, epoch_num=epoch_num, step_num=global_step)
                self.metric_logger.report_scalar("loss/current", loss, global_step)

                # Evaluate on batch if enabled
                if self.eval_on_batch and (global_step % self.eval_freq == 0):
                    current_metric = self.evaluate(dl_eval)
                    stop_criteria.update(epoch_num, current_metric)
                    should_stop = stop_criteria.should_stop()

                    self.save_checkpoint()
                    self.metric_logger.report_scalar(f"{self.metric_name}/best", self.best_metric, global_step)
                    self.metric_logger.report_scalar(f"{self.metric_name}/current", current_metric, global_step)
                    self.metric_logger.report_image("pert/current", self.pert_model.to_image(), global_step)

                batch_pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}" if loss is not None else "None",
                        f"{self.metric_name}_init": f"{self.init_metric:.4f}",
                        f"{self.metric_name}_best": f"{self.best_metric:.4f}",
                        f"{self.metric_name}_curr": f"{current_metric:.4f}",
                    }
                )

                global_step += 1

            batch_pbar.close()

            # Evaluate on epoch if enabled
            if not self.eval_on_batch and epoch_num % self.eval_freq == 0:
                current_metric = self.evaluate(dl_eval)
                stop_criteria.update(epoch_num, current_metric)
                should_stop = stop_criteria.should_stop()

                self.save_checkpoint()
                self.metric_logger.report_scalar(f"{self.metric_name}/best", self.best_metric, global_step)
                self.metric_logger.report_scalar(f"{self.metric_name}/current", current_metric, global_step)
                self.metric_logger.report_image("pert/current", self.pert_model.to_image(), global_step)

            epoch_pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}" if loss is not None else "None",
                    f"{self.metric_name}_init": f"{self.init_metric:.4f}",
                    f"{self.metric_name}_best": f"{self.best_metric:.4f}",
                    f"{self.metric_name}_curr": f"{current_metric:.4f}",
                }
            )

        epoch_pbar.close()

        # Update to best pert
        self.pert_model.set_pert(self.best_pert)

        # Final evaluation
        metrics = eval.full_analysis(self.pert_model, dl_eval, silent=False)
        self.metric_logger.report_globals(metrics)

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
            eval_freq (int): Every how many steps to evaluate the model performance.
            eval_on_batch (bool): If true, each batch is counted as step for evaluation, otherwise each epoch.
            metric_name (str): Name of the evaluation metric for display and logging.
            metric_func (Callable[[PertModule, Iterable[tuple[torch.Tensor, ...]]], float], optional): Function to compute the evaluation metric.
                If None, uses `eval.misclassification_rate`.
            log_dir (str, optional): Directory for storing logs. If None, logging is disabled.
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
    def evaluate(self, dl_eval: Iterable[tuple[torch.Tensor, ...]]) -> float:
        with self.autocast_context():
            return super().evaluate(dl_eval)
