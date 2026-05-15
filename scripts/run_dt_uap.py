from argparse import ArgumentParser
import sys
import pathlib
import torch


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.dt_uap import DT_UAP
from ulib import ActivationExtractor


class DT_UAP_Experiment(Experiment):
    """
    Official Arguments:
        - norm=inf
        - epsilon=10/255
        - targeted=True
        - train_batch=16
        - max_epochs=10
        - patience=10 (effectively no early stopping)

        - lr=0.01
        - weight_decay=1e-10
        - alpha_step=None
        - layer is last layer in the model
        - alpha is not specified
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        dt_args = parser.add_argument_group("DT-UAP Attack Arguments")

        dt_args.add_argument(
            "--lr",
            type=float,
            default=0.01,
            metavar="FLOAT",
            help="Learning rate for Adam optimization.",
        )

        dt_args.add_argument(
            "--weight_decay",
            type=float,
            default=1e-10,
            metavar="FLOAT",
            help="Weight decay for Adam optimizer.",
        )

        dt_args.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            metavar="FLOAT",
            help="The weight of the input images in the forward pass.",
        )

        dt_args.add_argument(
            "--alpha_step",
            type=float,
            default=None,
            metavar="FLOAT",
            help="The value of `alpha` will be linearly increased from 0 to `alpha`, with a step size of `alpha_step`.",
        )

        dt_args.add_argument(
            "--layer",
            type=str,
            default="model.avgpool",
            metavar="NAME",
            help="Layer name to extract output activations from",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()

        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        activ_extractor = ActivationExtractor(pert_model.model, args.layer, capture_output=False)

        return DT_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            alpha=args.alpha,
            alpha_step=args.alpha_step,
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            mixed_precision=args.use_amp.lower() == "true",
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    DT_UAP_Experiment().main()
