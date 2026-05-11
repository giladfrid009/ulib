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


class Uapgd_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("UAPGD Attack Arguments")

        args.add_argument(
            "--inner_eps",
            type=int,
            default=16,
            help="Epsilon for MIPGD (inner attack) in 255 scale.",
        )

        args.add_argument(
            "--inner_alpha",
            type=float,
            default=0.1,
            help="Alpha for MIPGD.",
        )

        args.add_argument(
            "--inner_gamma",
            type=float,
            default=0.0,
            help="Gamma for MIPGD.",
        )

        # ValueScheduler args
        args.add_argument(
            "--sched_init",
            type=float,
            default=0.1,
            help="Initial value for alpha scheduler.",
        )

        args.add_argument(
            "--sched_step",
            type=int,
            default=5,
            help="Step size for StepLR in alpha scheduler.",
        )

        args.add_argument(
            "--sched_gamma",
            type=float,
            default=0.2,
            help="Gamma for StepLR in alpha scheduler.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()

        eps = args.inner_eps / 255.0
        torch_attack = MIPGD(pert_model.model, eps=eps, alpha=args.inner_alpha, gamma=args.inner_gamma)

        alpha_sched = ValueScheduler(
            init_value=args.sched_init,
            sched_cls=StepLR,
            step_size=args.sched_step,
            gamma=args.sched_gamma,
        )

        return UAPGD(
            pert_model=pert_model,
            inner_attack=torch_attack,
            alpha_sched=alpha_sched,
            sched_on_batch=False,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    Uapgd_Experiment().main()
