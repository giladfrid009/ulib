# fix imports
import sys
import pathlib
from abc import ABC, abstractmethod
import argparse
import random
import time
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from ulib.evaluator import MetricEvaluator
from ulib.utils import env
from ulib import PertModule, StopCriteria, UnivAttack
from ulib.metric_logger import MetricLogger
from ulib.utils.logging import create_logger, setup_logging, loglevel_names

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset
from scripts.utils.load_evaluator import SUPPORTED_EVALUATORS, load_evaluator


logger = create_logger(__name__)


class Experiment(ABC):
    def __init__(self) -> None:
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""
        pass

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            choices=SUPPORTED_MODELS,
            default="resnet50",
            metavar="MODEL",
            help=f"The model name to attack. Available models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="imagenet",
            metavar="DATASET",
            help=f"The datasets to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--evaluator",
            type=str,
            choices=SUPPORTED_EVALUATORS,
            default="simple",
            metavar="EVAL",
            help=f"The attack evaluator to use. Available evaluators: {SUPPORTED_EVALUATORS}",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            metavar="NAME",
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--train_batch",
            type=int,
            default=50,
            metavar="SIZE",
            help="The training batch size.",
        )

        parser.add_argument(
            "--eval_batch",
            type=int,
            default=100,
            metavar="SIZE",
            help="The evaluation batch size.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=loglevel_names(),
            default="INFO",
            metavar="LEVEL",
            help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="If set, the experiment will run a quick test with reduced epochs and time.",
        )

        attack_args = parser.add_argument_group("Base attack parameters")

        attack_args.add_argument(
            "--eval_metric",
            type=str,
            default=None,
            metavar="NAME",
            help="The evaluation metric to use for selecting the best adversarial prompt. "
            "If not specified, the default metric of the first evaluator will be used.",
        )

        attack_args.add_argument(
            "--eval_freq",
            type=float,
            default=1,
            metavar="NUM",
            help="Frequency of evaluation during training, in epochs. Can be a float.",
        )

        attack_args.add_argument(
            "--use_amp",
            choices=["true", "false"],
            metavar="BOOL",
            default="false",
            help="Whether to use automatic mixed precision (AMP) for training.",
        )

        pert_args = parser.add_argument_group("Perturbation parameters")

        pert_args.add_argument(
            "--epsilon",
            type=int,
            default=16,
            metavar="EPS",
            help="The maximum perturbation magnitude eps/255 for the chosen norm.",
        )

        pert_args.add_argument(
            "--norm",
            type=float,
            default=float("inf"),
            choices=[1, 2, float("inf")],
            metavar="NORM",
            help="The perturbation norm constraint (1, 2, or inf).",
        )

        pert_args.add_argument(
            "--random_init",
            action="store_true",
            help="If set, the perturbation will be randomly initialized within the epsilon-ball.",
        )

        stop_args = parser.add_argument_group("Stopping Criteria")

        stop_args.add_argument(
            "--max_time",
            type=int,
            default=20,
            metavar="MINUTES",
            help="The maximum training time in minutes.",
        )

        stop_args.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            metavar="NUM",
            help="The maximum number of training epochs.",
        )

        stop_args.add_argument(
            "--patience",
            type=int,
            default=20,
            metavar="NUM",
            help="Early stopping patience in number of evaluations.",
        )

        self.add_arguments(parser)
        self._parsed_args = parser.parse_args()
        args = self.args()

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()

        return args

    def prepare_environment(self, seed: int | None):
        if seed is None:
            seed = random.randint(0, 10000)
        logger.info(f"Random seed: {seed}")
        env.set_seed(seed)

    @abstractmethod
    def initialize_attack(
        self,
        pert_model: PertModule,
        evaluator: MetricEvaluator,
        eval_freq: float,
        mixed_precision: bool,
        metric_logger: MetricLogger,
    ) -> UnivAttack:
        pass

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        with MetricLogger(
            args.model,
            args.dataset,
            f"L_{args.norm}",
            f"Eps_{args.epsilon}",
            args.run_name,
            root_dir="logs",
            project="CLF-IML",
            disabled=args.test_run,
        ) as metric_logger:
            logger.info(f"Loading model: {args.model}")
            model, ds_transform = load_model(args.model, args.dataset, device="cuda:0")

            logger.info(f"Loading dataset: {args.dataset}")
            dl_train, dl_eval, dl_test = load_dataset(args.dataset, ds_transform, args.train_batch, args.eval_batch)
            logger.info(
                f"Loaded datasets with sample counts: "
                f"(train, val, test) = ({dl_train.total}, {dl_eval.total}, {dl_test.total})."
            )

            pert_model = PertModule(
                model,
                data_shape=dl_train.get_tensor(0).shape[1:],
                eps=args.epsilon / 255.0,
                norm=args.norm,
                random_init=args.random_init,
            )

            evaluator = load_evaluator(args.evaluator, pert_model, main_metric=args.eval_metric)

            logger.info("Initializing attack...")
            univ_attack = self.initialize_attack(
                pert_model=pert_model,
                evaluator=evaluator,
                eval_freq=args.eval_freq,
                mixed_precision=args.use_amp.lower() == "true",
                metric_logger=metric_logger,
            )

            metric_logger.set_tags(
                attack=type(univ_attack).__name__,
                model=type(model).__name__,
                dataset=args.dataset,
                norm=pert_model.norm,
                eps=f"{args.epsilon}",
            )

            if main_file := getattr(sys.modules.get("__main__"), "__file__", None):
                metric_logger.upload_code(main_file)
            if expr_file := getattr(sys.modules.get(__name__), "__file__", None):
                metric_logger.upload_code(expr_file)

            logger.info("Running attack...")

            stop = StopCriteria(
                max_epochs=args.max_epochs if not args.test_run else 1,
                max_time=args.max_time * 60 if not args.test_run else 5,
                patience=args.patience,
                max_evals=None if not args.test_run else 1,
            )

            pert_model = univ_attack.fit(dl_train, dl_eval, stop_criteria=stop)

            logger.info("Running test evaluation...")
            metrics = univ_attack.evaluate(dl_test)
            metric_logger.report_globals(metrics)

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
