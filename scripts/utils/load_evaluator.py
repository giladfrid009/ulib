from enum import Enum
from ulib.utils.logging import create_logger
from ulib.evaluator import MetricEvaluator, SimpleEvaluator, ExtendedEvaluator
from ulib import PertModule

logger = create_logger(__name__)


class EvalName(str, Enum):
    SIMPLE = "simple"
    EXTENDED = "extended"


SUPPORTED_EVALUATORS = [e.value for e in EvalName]


def load_evaluator(
    name: str,
    pert_module: PertModule,
    main_metric: str | None = None,
) -> MetricEvaluator:
    if name == EvalName.SIMPLE:
        return SimpleEvaluator(pert_module, main_metric=main_metric, verbose=True)

    if name == EvalName.EXTENDED:
        return ExtendedEvaluator(pert_module, main_metric=main_metric, verbose=True)

    raise ValueError(f"Unsupported evaluator name: {name}.")
