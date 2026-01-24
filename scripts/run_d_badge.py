from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.d_badge import D_BADGE


class D_BADGE_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("D-BADGE Attack Arguments")
        
        args.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="Learning rate for Adam optimizer",
        )
        
        args.add_argument(
            "--delta",
            type=float,
            default=0.01,
            help="L-inf radius of the random noise at which the gradient is estimated.",
        )
        
        args.add_argument(
            "--delta_decay",
            type=float,
            default=0.9,
            help="Factor by which to decay delta each epoch.",
        )
        
        args.add_argument(
            "--gamma",
            type=float,
            default=1.0,
            help="Scaling factor for the gradient.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)

        return D_BADGE(
            pert_model=pert_model,
            optimizer=optimizer,
            delta=args.delta,
            delta_decay=args.delta_decay,
            gamma=args.gamma,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    D_BADGE_Experiment().main()
