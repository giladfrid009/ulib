# fix imports
import sys
import pathlib
import argparse
import random
import torch
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from ulib.utils import env
from ulib import PertModule
from ulib.utils.logging import create_logger, setup_logging, loglevel_names

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset
from scripts.utils.load_evaluator import SUPPORTED_EVALUATORS, load_evaluator


logger = create_logger(__name__)


class Evaluate:
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
            "pert_path",
            type=str,
            metavar="PATH",
            help="Path to the perturbation .pt file to evaluate.",
        )

        parser.add_argument(
            "--models",
            type=str,
            choices=SUPPORTED_MODELS,
            nargs="+",
            default=SUPPORTED_MODELS,
            metavar="MODEL",
            help=f"The model names to attack. Available models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="imagenet",
            metavar="DATASET",
            help=f"The dataset to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--evaluator",
            type=str,
            choices=SUPPORTED_EVALUATORS,
            default="extended",
            metavar="EVAL",
            help=f"The attack evaluator to use. Available evaluators: {SUPPORTED_EVALUATORS}",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            metavar="SIZE",
            help="The evaluation batch size.",
        )

        parser.add_argument(
            "--include_train",
            action="store_true",
            help="Whether to evaluate on the training set as well.",
        )

        parser.add_argument(
            "--include_eval",
            action="store_true",
            help="Whether to evaluate on the evaluation set as well.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
        )

        parser.add_argument(
            "--name_format",
            type=str,
            metavar="FMT",
            default="{model}_{dataset}_{split}.csv",
            help="Format string for naming the results files. Must include [{model}, {dataset}, {split}] placeholders.",
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Whether to overwrite existing results files. If not set, the script will raise an error if a results file already exists for a given model/dataset/split.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=loglevel_names(),
            default="INFO",
            metavar="LEVEL",
            help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
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

    def results_path(self, dataset: str, model: str, split: str) -> str:
        args = self.args()
        folder = pathlib.Path(args.pert_path).parent / "evaluations"

        model_name = model.split("/")[-1].lower()
        dataset_name = dataset.lower()
        file_name = args.name_format.format(model=model_name, dataset=dataset_name, split=split)
        full_path: pathlib.Path = folder / file_name

        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created results directory at: {full_path.parent.as_posix()}")

        if full_path.exists() and not args.overwrite:
            raise FileExistsError(f"Results file already exists: {full_path.as_posix()}")

        return full_path.as_posix()

    def save_results(self, results: dict[str, float], model_name: str, split_name: str):
        results_path = self.results_path(self.args().dataset, model_name, split_name)
        df = pd.DataFrame([results])
        df.to_csv(results_path, index=False)
        logger.info(f"Saved {split_name} results to: {results_path}")

    def load_pert(self, pert_module: PertModule) -> PertModule:
        pert_path = self.args().pert_path
        pert_weights = torch.load(pert_path, map_location=pert_module.device, weights_only=True)
        expected_shape = (1,) + pert_module.data_shape  # (1, C, H, W) is the expected shape

        if expected_shape[:2] != pert_weights.shape[:2]:
            raise ValueError(f"Expected perturbation to have shape (1, C, ...), but got {pert_weights.shape}.")

        if pert_weights.ndim != len(expected_shape):
            raise ValueError(f"Unexpected number of dimensions in weights: {pert_weights.shape}.")

        if pert_weights.shape[1:] != pert_module.data_shape:
            # resize the perturbation weights to match the expected data shape
            logger.warning(f"Resizing perturbation: {pert_weights.shape} --> {expected_shape}.")
            pert_weights = torch.nn.functional.interpolate(pert_weights, size=expected_shape[2:], mode="bilinear", align_corners=False, antialias=True)

        pert_module.set_pert(pert_weights)
        return pert_module

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        for model_name in args.models:
            logger.info(f"Loading model: {model_name}")
            model, ds_transform = load_model(model_name, args.dataset, device="cuda:0")

            logger.info(f"Loading dataset: {args.dataset}")
            dl_train, dl_eval, dl_test = load_dataset(args.dataset, ds_transform, args.batch_size, args.batch_size)
            logger.info(f"Loaded datasets with sample counts: (train, val, test) = ({dl_train.total}, {dl_eval.total}, {dl_test.total}).")

            pert_model = PertModule(
                model,
                data_shape=dl_train.get_tensor(0).shape[1:],
                eps=args.epsilon / 255.0,
                norm=args.norm,
            )

            pert_model = self.load_pert(pert_model)
            evaluator = load_evaluator(args.evaluator, pert_model)

            if args.include_train:
                logger.info("Evaluating on training set...")
                train_metrics = evaluator.evaluate(dl_train)
                self.save_results(train_metrics, model_name, "train")

            if args.include_eval:
                logger.info("Evaluating on evaluation set...")
                eval_metrics = evaluator.evaluate(dl_eval)
                self.save_results(eval_metrics, model_name, "eval")

            logger.info("Evaluating on test set...")
            test_metrics = evaluator.evaluate(dl_test)
            self.save_results(test_metrics, model_name, "test")

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user.")
            sys.exit(0)


if __name__ == "__main__":
    Evaluate().main()
