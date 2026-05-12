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

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        activ_extractor = ActivationExtractor(pert_model.model, *args.layers, capture_output=True)

        return DT_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            alpha_step_size=args.alpha_step_size,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    DT_UAP_Experiment().main()
