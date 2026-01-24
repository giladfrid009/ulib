from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.gd_uap import GD_UAP


class GD_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("GD-UAP Attack Arguments")
        
        args.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate for Adam optimizer",
        )
        
        args.add_argument(
            "--sat_thresh",
            type=float,
            default=0.5,
            help="Saturation threshold.",
        )
        
        args.add_argument(
            "--sat_delta",
            type=float,
            default=1e-5,
            help="Saturation delta.",
        )
        
        args.add_argument(
            "--data_dependant",
            choices=["true", "false"],
            default="true",
            help="Whether the attack is data dependent.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)

        return GD_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            data_dependant=(args.data_dependant == "true"),
            sat_thresh=args.sat_thresh,
            sat_delta=args.sat_delta,
            # general args
            mixed_precision=mixed_precision,
            eval_freq=eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    GD_Experiment().main()
