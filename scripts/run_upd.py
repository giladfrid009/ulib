from argparse import ArgumentParser
import sys
import pathlib
import torch
import torchattacks.attack


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib.attacks.upd import UPD, CosSim
from ulib.activation_extractor import ActivationExtractor
import sample_attacks


INNER_ATTACKS = [
    "fgsm",
    "ffgsm",
    "pgd",
    "cw",
    "tpgd",
    "apgd",
    "jitter",
    "ila",
]


class UPD_Experiment(Experiment):
    """
    Official Arguments (from paper):
        - norm=inf
        - epsilon=16/255
        - train_batch=128
        - max_epochs=30
        
        - inner_attack=fgsm
        - inner_steps=16/255
        - inner_steps=5
        - lr=0.02
        - skip_fooled=true
        - skip_failed=true
        - layers is last model layer
    """
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        upd_args = parser.add_argument_group("UPD Attack Arguments")

        upd_args.add_argument(
            "--inner_attack",
            type=str,
            default="fgsm",
            choices=INNER_ATTACKS,
            help="The inner attack method to use for crafting the universal perturbation.",
        )

        upd_args.add_argument(
            "--inner_epsilon",
            type=float,
            default=None,
            help="The maximum epsilon value for the inner attack. If not specified, then using the same epsilon as the outer attack.",
        )

        upd_args.add_argument(
            "--inner_steps",
            type=int,
            default=5,
            help="The number of steps for the inner attack (if applicable).",
        )

        upd_args.add_argument(
            "--lr",
            type=float,
            default=2e-2,
            metavar="FLOAT",
            help="Learning rate for the adversarial trigger optimization.",
        )

        upd_args.add_argument(
            "--layers",
            type=str,
            nargs="+",
            default=["model.avgpool"],
            metavar="NAMES",
            help="Names of the layers to extract activations from.",
        )

        upd_args.add_argument(
            "--skip_fooled",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that are already fooled by the universal trigger.",
        )

        upd_args.add_argument(
            "--skip_failed",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that the inner attack failed to attack.",
        )

    def initialize_attack(self, pert_model, evaluator, dl_train, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)
        criterion = CosSim(reduce_fn=lambda losses: torch.sum(losses, dim=-1).mean())
        activ_extractor = ActivationExtractor(pert_model.model, *args.layers, capture_output=True)

        def attack_builder(model: torch.nn.Module, epoch: int) -> torchattacks.attack.Attack:

            # compute the current epsilon value based on the epoch number (linear schedule)
            max_eps = args.inner_epsilon if args.inner_epsilon is not None else pert_model.eps
            eps = (max_eps / args.max_epochs) * (epoch + 1)

            if args.inner_attack == "fgsm":
                return torchattacks.FGSM(model, eps=eps)
            if args.inner_attack == "ffgsm":
                return torchattacks.FFGSM(model, eps=eps, alpha=eps)
            if args.inner_attack == "pgd":
                return torchattacks.PGD(model, eps=eps, alpha=eps / 3, steps=args.inner_steps)
            if args.inner_attack == "cw":
                return torchattacks.CW(model, lr=eps / 3, steps=args.inner_steps)
            if args.inner_attack == "tpgd":
                return torchattacks.TPGD(model, eps=eps, alpha=eps / 3, steps=args.inner_steps)
            if args.inner_attack == "apgd":
                return torchattacks.APGD(model, eps=eps, steps=args.inner_steps, loss="dlr")
            if args.inner_attack == "jitter":
                return torchattacks.Jitter(model, eps=eps, alpha=eps / 3, steps=args.inner_steps)
            if args.inner_attack == "ila":
                return sample_attacks.AttackWrapper(sample_attacks.ILA(model, epsilon=eps, epoch=args.inner_steps, alpha=eps / args.inner_steps))

            raise ValueError(f"Unsupported inner attack method: {args.inner_attack}")

        return UPD(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            inner_attack=attack_builder,
            activ_extractor=activ_extractor,
            skip_already_fooled=args.skip_fooled.lower() == "true",
            skip_failed_attacks=args.skip_failed.lower() == "true",
            # general args
            eval_freq=args.eval_freq,
            evaluator=evaluator,
            mixed_precision=args.use_amp.lower() == "true",
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    UPD_Experiment().main()
