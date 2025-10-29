import pathlib
from clearml import Task
from typing import Any
import numpy as np
import PIL.Image
import torch
from ulib.utils.logging import create_logger


logger = create_logger(__name__)


class MetricLogger:
    def __init__(
        self,
        *names: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
    ):
        if len(names) == 0 and not disabled:
            raise ValueError("At least one name component must be provided.")

        self.run_name = str.join(" - ", names)
        self.project = project
        self.root_dir = root_dir
        self.disabled = disabled

        self.log_dir: str | None = None
        self.cm_task: Task | None = None

        if not disabled:
            Task.set_random_seed(None)  # NOTE: are you kidding me
            self.log_dir = self._create_directory(root_dir, *names)
            self.cm_task = Task.init(project_name=project, task_name=self.run_name)

    def get_hparams(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "project": self.project,
            "root_dir": self.root_dir,
            "log_dir": self.log_dir,
            "disabled": self.disabled,
        }

    def _create_directory(self, *subdir_parts: str) -> str:
        log_path = pathlib.Path(*subdir_parts)
        log_path.mkdir(parents=True, exist_ok=True)
        path = log_path.as_posix()
        logger.info(f"Created log directory at: {path}")
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.cm_task is not None:
            self.cm_task.flush(wait_for_uploads=True)
            self.cm_task.close()
            self.cm_task = None  # type: ignore

    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        """
        Logs hyperparameters under the key: `category/<param_name>`.
        If a parameter is a dictionary, it is flattened.

        Args:
            category (str): The category under which to log the hyperparameters.
            *args (dict): Positional dictionaries of hyperparameters to log.
            **kwargs: Keyword arguments of hyperparameters to log.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        hparams = flatten_dict(kwargs, category)
        for d in args:
            hparams.update(flatten_dict(d, category))

        self.cm_task.update_parameters(hparams)

    def upload_code(self, file_path: str):
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if not pathlib.Path(file_path).is_file():
            logger.warning(f"File '{file_path}' does not exist. Cannot log code.")
            return

        file_name = pathlib.Path(file_path).name
        file_content = pathlib.Path(file_path).read_text()
        upload_result = self.cm_task.upload_artifact(
            name=file_name,
            artifact_object=file_path,
            preview=file_content,
            metadata={"type": "code", "full_path": file_path},
        )

        if not upload_result:
            logger.warning(f"Failed to log code file '{file_path}'.")

    def set_tags(self, **tags):
        """
        Sets tags for the current experiment.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if len(tags) == 0:
            logger.warning("No tags provided.")
            return

        tags = {k.title(): str(v) for k, v in tags.items()}
        formatted = [f"{tag}: {text}" for tag, text in tags.items()]
        self.cm_task.add_tags(formatted)

    def report_globals(self, metrics: dict[str, int | float]):
        """
        Logs a dictionary of final, global run metrics.
        For example, used to log final evaluation metrics.

        Args:
            metrics (dict[str, int | float]): The metrics to log.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        for k, v in metrics.items():
            self.cm_task.logger.report_single_value(k, v)

    def report_scalars(self, scalers: dict[str, int | float], step: int):
        """
        Logs multiple scalar values.

        Args:
            scalers (dict[str, int | float]): The scalar values to log.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        for key, value in scalers.items():
            self.report_scalar(key, value, step)

    def report_scalar(self, tag: str, value: int | float | None, step: int):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar value.
            value (int | float | None): The scalar value to log.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()

        if value is None:
            value = float("nan")

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_scalar(title=title, series=series, value=float(value), iteration=step)

    def report_image(self, tag: str, image: torch.Tensor | np.ndarray | PIL.Image.Image, step: int):
        """
        Logs an image.

        Args:
            tag (str): The name of the image.
            image (torch.Tensor | np.ndarray | PIL.Image.Image): The image to log, RGB format.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()
        series = series.title()

        if isinstance(image, torch.Tensor):
            image = image.numpy(force=True)

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_image(title=title, series=series, image=image, iteration=step, max_image_history=-1)
