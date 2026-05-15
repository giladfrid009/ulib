from argparse import ArgumentParser
import sys
import pathlib
import torch


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.df_uap import DF_UAP


class DF_Experiment(Experiment):
    """
    Official Arguments:
        - norm=inf
        - epsilon=10/255
        - targeted=True
        - train_batch=32
        - max_steps=1000
        - patience=1000 (effectively no early stopping)
        - lr=0.005
        - kappa is not specified
    """
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        df_args = parser.add_argument_group("DF-UAP Attack Arguments")
        
        df_args.add_argument(
            "--lr",
            type=float,
            default=0.005,
            metavar="FLOAT",
            help="Learning rate for Adam optimizer",
        )
        
        df_args.add_argument(
            "--kappa",
            type=float,
            default=0.0,
            metavar="FLOAT",
            help="Kappa value for DF-UAP attack",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)

        return DF_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            kappa=args.kappa,
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            mixed_precision=args.use_amp.lower() == "true",
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    DF_Experiment().main()
