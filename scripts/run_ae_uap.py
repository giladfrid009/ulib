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
    """
    Official Arguments: (data free) (from paper)
        - norm=inf
        - epsilon=10/255
        - max_steps=1000
        - train_batch=32

        - data_free=True
        - lr=0.005
        - cr_lambda=0.05
        - inner_epsilon=20/255
        - inner_epsilon_start=14/255
        - inner_epsilon_zoom=20

    Official Arguments: (data dependent) (from paper)
        - norm=inf
        - epsilon=10/255
        - max_steps=1000
        - train_batch=32

        - data_free=False
        - lr=0.005
        - cr_lambda=0.05
        - inner_epsilon=4/255
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        ae_args = parser.add_argument_group("AE-UAP Attack Arguments")

        ae_args.add_argument(
            "--lr",
            type=float,
            default=0.005,
            metavar="FLOAT",
            help="Learning rate for optimizing the universal perturbation.",
        )

        ae_args.add_argument(
            "--inner_epsilon",
            type=float,
            default=20 / 255,
            metavar="FLOAT",
            help="Epsilon for the inner attack (epsilon_2 in paper).",
        )

        ae_args.add_argument(
            "--inner_epsilon_start",
            type=float,
            default=None,
            metavar="FLOAT",
            help="Starting epsilon for the inner attack (a_0 in paper). If not specified, no scheduling is applied.",
        )

        ae_args.add_argument(
            "--inner_epsilon_zoom",
            type=float,
            default=1.0,
            metavar="FLOAT",
            help="Zoom factor for the inner attack epsilon scheduling (alpha in Eq.13).",
        )

        ae_args.add_argument(
            "--inner_steps",
            type=int,
            default=10,
            metavar="INT",
            help="Number of steps for the inner attack (AE_MIFGSM).",
        )

        ae_args.add_argument(
            "--inner_decay",
            type=float,
            default=1.0,
            metavar="FLOAT",
            help="Decay factor for the inner attack (mu in paper).",
        )

        ae_args.add_argument(
            "--data_free",
            action="store_true",
            help="Whether to use the data-free version of the attack.",
        )

        ae_args.add_argument(
            "--cr_lambda",
            type=float,
            default=0.05,
            metavar="FLOAT",
            help="Regularization strength for the consistency loss term (lambda in paper).",
        )

        ae_args.add_argument(
            "--noise_scale",
            type=float,
            default=16 / 255,
            metavar="FLOAT",
            help="Uniform noise scale for the data-free version of the attack.",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)

        # compute curriculum scheduling parameters
        max_steps = args.max_steps
        if max_steps is None:
            max_steps = args.max_epochs * len(dl_train)

        eps_start = args.inner_epsilon_start
        if eps_start is None:
            eps_start = args.inner_epsilon

        eps_max = args.inner_epsilon
        zoom_factor = args.inner_epsilon_zoom

        assert zoom_factor >= 1.0, "inner_epsilon_zoom must be >= 1.0 to ensure a valid scheduling of inner attack epsilon values."
        assert eps_start <= eps_max, "inner_epsilon_start must be less than or equal to inner_epsilon."

        def attack_builder(model: torch.nn.Module, step: int) -> AE_MIFGSM:
            # curriculum scheduling for inner attack epsilon (Eq.13 in paper)
            eps = eps_start + zoom_factor * (step / max_steps) * (eps_max - eps_start)
            eps = min(eps, eps_max)  # ensure that epsilon does not exceed the specified maximum

            return AE_MIFGSM(
                model=model,
                eps=eps,
                steps=args.inner_steps,
                decay=args.inner_decay,
            )

        return AE_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            inner_attack=attack_builder,
            data_free=args.data_free,
            cr_lambda=args.cr_lambda,
            noise_scale=args.noise_scale,
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            mixed_precision=args.use_amp.lower() == "true",
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    AE_Experiment().main()
