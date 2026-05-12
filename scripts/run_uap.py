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
from ulib.attacks.uap import UAP
import sample_attacks

INNER_ATTACKS = [
    "deepfool",
    "fgsm",
    "ffgsm",
    "pgd",
    "cw",
    "tpgd",
    "apgd",
    "jitter",
    "ila",
]


class UAP_Experiment(Experiment):
    """
    Official Arguments (untargeted) (from paper and official codebase):
        - norm=inf
        - epsilon=10/255
        - targeted=false
        - train_batch=1
        
        - inner_attack=deepfool
        - inner_steps=10
        
        - evaluator=extended
        - eval_metric=FR
        - target_value=0.8
        - max_epochs=1000 (effectively no epoch limit)
        - patience=1000 (effectively no early stopping)
        
    Official Arguments (targeted) (from paper and official codebase): [https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/targeted_universal_perturbation.py]
        - norm=inf
        - epsilon=10/255
        - targeted=true
        - train_batch=1
        
        - inner_attack=fgsm
        
        - evaluator=simple
        - eval_metric=ACC
        - target_value=0.8
        - max_epochs=20
        - patience=20 (effectively no early stopping)
    """
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        uap_args = parser.add_argument_group("UAP Attack Arguments")

        uap_args.add_argument(
            "--inner_attack",
            type=str,
            default="fgsm",
            choices=INNER_ATTACKS,
            help="The inner attack method to use for crafting the universal perturbation.",
        )

        uap_args.add_argument(
            "--inner_epsilon",
            type=float,
            default=None,
            help="The maximum epsilon value for the inner attack. If not specified, then using the same epsilon as the outer attack.",
        )

        uap_args.add_argument(
            "--inner_steps",
            type=int,
            default=5,
            help="The number of steps for the inner attack (if applicable).",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()

        def attack_builder(model: torch.nn.Module, epoch: int) -> torchattacks.attack.Attack:

            eps = args.inner_epsilon if args.inner_epsilon is not None else pert_model.eps

            if args.inner_attack == "deepfool":
                return torchattacks.DeepFool(model, steps=args.inner_steps, overshoot=0.2)
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

        return UAP(
            pert_model=pert_model,
            inner_attack=attack_builder,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
            targeted=self.targeted,
        )


if __name__ == "__main__":
    UAP_Experiment().main()
