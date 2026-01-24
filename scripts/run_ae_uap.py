from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.ae_uap import AE_UAP, AE_MIFGSM


class AE_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        ae_args = parser.add_argument_group("AE-UAP Attack Arguments")
        
        ae_args.add_argument(
            "--lr",
            type=float,
            default=0.005,
            metavar="FLOAT",
            help="Learning rate for Adam optimizer",
        )
        
        ae_args.add_argument(
            "--gamma",
            type=float,
            default=0.5,
            help="Gamma parameter for AE_UAP",
        )
        
        ae_args.add_argument(
            "--inner_eps",
            type=float,
            default=4/255,
            help="Epsilon for inner attack (AE_MIFGSM)",
        )
        
        ae_args.add_argument(
            "--inner_steps",
            type=int,
            default=10,
            help="Steps for inner attack",
        )
        
        ae_args.add_argument(
            "--inner_decay",
            type=float,
            default=1.0,
            help="Decay for inner attack",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)
        
        torch_attack = AE_MIFGSM(
            pert_model.model, 
            eps=args.inner_eps, 
            alpha=args.inner_eps / 10,  # Following notebook pattern
            steps=args.inner_steps, 
            decay=args.inner_decay
        )

        return AE_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            inner_attack=torch_attack,
            gamma=args.gamma,
            # general args
            mixed_precision=mixed_precision,
            eval_freq=eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    AE_Experiment().main()
