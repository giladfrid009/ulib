from argparse import ArgumentParser
import sys
import pathlib
import torch


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.dm_uap import DM_UAP


class DM_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        dm_args = parser.add_argument_group("DM-UAP Attack Arguments")

        dm_args.add_argument(
            "--beta",
            type=float,
            default=9.0,
            metavar="FLOAT",
            help="Loss clamping value (beta in paper).",
        )

        dm_args.add_argument(
            "--theta_neighborhood",
            type=float,
            default=1.0,
            metavar="FLOAT",
            help="Model maximum neighborhood size (Rho in paper).",
        )

        dm_args.add_argument(
            "--data_neighborhood",
            type=float,
            default=32.0,
            metavar="FLOAT",
            help="Data maximum neighborhood size (r in paper).",
        )

        dm_args.add_argument(
            "--data_steps",
            type=int,
            default=10,
            metavar="INT",
            help="Number of data optimization steps in the inner PGD attack (Kd in paper).",
        )

        dm_args.add_argument(
            "--theta_steps",
            type=int,
            default=10,
            metavar="INT",
            help="Number of steps of model param optimization in the inner PGD attack (Km in paper).",
        )

        dm_args.add_argument(
            "--smooth_rate",
            type=float,
            default=0.2,
            metavar="FLOAT",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()

        return DM_UAP(
            pert_model=pert_model,
            total_epochs=args.max_epochs,
            beta=args.beta,
            theta_neighborhood=args.theta_neighborhood,
            data_neighborhood=args.data_neighborhood,
            data_steps=args.data_steps,
            theta_steps=args.theta_steps,
            smooth_rate=args.smooth_rate,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    DM_Experiment().main()
