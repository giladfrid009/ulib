from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.fff import FFF


class FFF_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("FFF Attack Arguments")
        
        args.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate for Adam optimizer",
        )
        
        args.add_argument(
            "--divide_every",
            type=float,
            default=50,
            help="Divide the perturbation by 2 every `divide_every` epochs. Set to 0 to disable.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)

        return FFF(
            pert_model=pert_model,
            optimizer=optimizer,
            divide_every=args.divide_every,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    FFF_Experiment().main()
