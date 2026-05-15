from argparse import ArgumentParser
import sys
import pathlib
from torch.optim.lr_scheduler import StepLR

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.uapgd import MIPGD, UAPGD, ValueScheduler


class UAPGD_Experiment(Experiment):
    """
    Official Arguments (from paper):
        - norm=inf
        - epsilon=10/255
        - train_batch=1
        - max_epochs=20
        - patience=20 (effectively no early stopping)

        - inner_beta=0.5
        - inner_gamma=1e-5
        - inner_steps=10

        - for VGG and ResNet:
            - inner_alpha=1.0
            - sched_decay=0.1
            - sched_period=5  (decrease at epochs 5, 10)

        - for Inception:
            - inner_alpha=0.5
            - sched_decay=0.2
            - sched_period=5  (decrease at epochs 5, 10, 15)
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        uapgd_args = parser.add_argument_group("UAPGD Attack Arguments")

        uapgd_args.add_argument(
            "--inner_alpha",
            type=float,
            default=0.1,
            help="Initial step size for MIPGD (inner attack).",
        )

        uapgd_args.add_argument(
            "--inner_beta",
            type=float,
            default=0.5,
            help="Momentum factor for MIPGD (inner attack).",
        )

        uapgd_args.add_argument(
            "--inner_gamma",
            type=float,
            default=0.0,
            help="Weight gain for MIPGD (inner attack).",
        )

        uapgd_args.add_argument(
            "--inner_steps",
            type=int,
            default=10,
            help="Number of steps for MIPGD (inner attack).",
        )

        uapgd_args.add_argument(
            "--sched_period",
            type=int,
            default=5,
            help="Period of learning rate decay for StepLR in alpha scheduler.",
        )

        uapgd_args.add_argument(
            "--sched_decay",
            type=float,
            default=0.2,
            help="Multiplicative factor of learning rate decay for StepLR in alpha scheduler.",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()

        torch_attack = MIPGD(
            pert_model.model,
            eps=pert_model.eps,
            alpha=args.inner_alpha,
            beta=args.inner_beta,
            gamma=args.inner_gamma,
            steps=args.inner_step,
        )

        alpha_sched = ValueScheduler(
            init_value=args.inner_alpha,
            sched_cls=StepLR,
            step_size=args.sched_period,
            gamma=args.sched_decay,
        )

        return UAPGD(
            pert_model=pert_model,
            inner_attack=torch_attack,
            alpha_sched=alpha_sched,
            sched_on_batch=False,
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    UAPGD_Experiment().main()
