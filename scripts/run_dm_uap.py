from argparse import ArgumentParser
import sys
import pathlib
import torch


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.dm_uap import DM_UAP, PGD_Theta_Apart


class DM_Experiment(Experiment):
    """
    Official Arguments: (from paper and official codebase)
        - norm=inf
        - epsilon=10/255
        - train_batch=125
        - max_epochs=20
        - patience=20 (effectively no early stopping)

        - lr=0.1
        - beta=9
        - smooth_rate=0.2
        - data_neighborhood=32
        - data_steps=10
        - theta_neighborhood=1
        - theta_steps=10
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        dm_args = parser.add_argument_group("DM-UAP Attack Arguments")

        dm_args.add_argument(
            "--lr",
            type=float,
            default=0.1,
            metavar="FLOAT",
            help="Learning rate for the attack optimization.",
        )

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
            help="The smoothing rate used in the GLS loss. Could go either positive or negative.",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()
        optimizer = torch.optim.AdamW(pert_model.parameters(), lr=args.lr)

        # curriculum scheduling parameters
        total_epochs = args.max_epochs
        rho_step = args.theta_neighborhood / total_epochs
        eps_step = args.data_neighborhood / total_epochs

        def attack_builder(model: torch.nn.Module, epoch_num: int) -> PGD_Theta_Apart:
            rho_current = rho_step * (epoch_num + 1)
            eps_current = eps_step * (epoch_num + 1)

            return PGD_Theta_Apart(
                model=model,
                eps=eps_current,
                rho=rho_current,
                data_steps=args.data_steps,
                theta_steps=args.theta_steps,
                random_start=False,
                eps_for_division=1e-10,
                smooth_rate=args.smooth_rate,
            )

        return DM_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            inner_attack=attack_builder,
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    DM_Experiment().main()
