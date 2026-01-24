from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.usgd import USGD


class USGD_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("USGD Attack Arguments")
        
        args.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="Learning rate for Adam optimizer",
        )
        
        args.add_argument(
            "--skip_fooled",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip already fooled samples during attack.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr, maximize=True)
        criterion = torch.nn.CrossEntropyLoss()

        return USGD(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            skip_already_fooled=(args.skip_fooled.lower() == "true"),
            # general args
            mixed_precision=mixed_precision,
            eval_freq=eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    USGD_Experiment().main()
