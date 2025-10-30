import time
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

    def update(self, epoch: int, value: float | None) -> None:
        """Update internal state with new metrics."""
        self._epoch = epoch
        self._total_evals += 1

        if value is not None:
            if (value - self._best_value) >= self.patience_delta:
                self._best_value = value
                self._patience_counter = 0
            else:
                self._patience_counter += 1

    def should_stop(self) -> bool:
        """Check if any stopping condition is met."""
        if self.target_value is not None and self._best_value >= self.target_value:
            logger.info(f"Stopping: Target value reached :: ({self.target_value})")
            return True

        if self._epoch >= self.max_epochs:
            logger.info(f"Stopping: Max epochs reached :: ({self.max_epochs})")
            return True

        if self.max_evals is not None and self._total_evals >= self.max_evals:
            logger.info(f"Stopping: Max evals reached :: ({self.max_evals})")
            return True

        if self.patience is not None and self._patience_counter >= self.patience:
            logger.info(f"Stopping: Patience exceeded :: ({self.patience})")
            return True

        if self.max_time is not None and (time.time() - self._start_time) > self.max_time:
            logger.info(f"Stopping: Max time reached :: ({self.max_time} sec)")
            return True

        return False
