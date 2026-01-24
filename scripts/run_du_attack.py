from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.du_attack import DU_ATTACK


class Du_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("DU-Attack Arguments")
        
        args.add_argument(
            "--alpha",
            type=float,
            default=0.2,
            help="Learning rate for the perturbation.",
        )
        
        args.add_argument(
            "--beta",
            type=float,
            default=0.9,
            help="Momentum factor.",
        )
        
        args.add_argument(
            "--steps",
            type=int,
            default=30,
            help="Number of steps to take in the optimization process at each batch.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        
        return DU_ATTACK(
            pert_model=pert_model,
            alpha=args.alpha,
            beta=args.beta,
            steps=args.steps,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    Du_Experiment().main()
