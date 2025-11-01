from argparse import ArgumentParser
import sys
import pathlib
import torch


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.fg_uap import FG_UAP
from ulib.activation_extractor import ActivationExtractor


class FG_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        fg_args = parser.add_argument_group("FG-UAP Attack Arguments")
        fg_args.add_argument(
            "--lr",
            type=float,
            default=0.01,
            metavar="FLOAT",
            help="Learning rate for Adam optimizer",
        )
        fg_args.add_argument(
            "--weight-decay",
            type=float,
            default=1e-5,
            metavar="FLOAT",
            help="Weight decay for Adam optimizer",
        )
        fg_args.add_argument(
            "--layer-name",
            type=str,
            default="model.avgpool",
            metavar="NAME",
            help="Layer name to extract output activations from",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_logger):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        activ_extractor = ActivationExtractor(pert_model.model, args.layer_name, exact_match=True)

        return FG_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_logger=metric_logger,
        )


if __name__ == "__main__":
    FG_Experiment().main()
