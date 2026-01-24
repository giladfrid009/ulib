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
from ulib.attacks.iml_uap import IML_UAP, CosSim
from ulib.activation_extractor import ActivationExtractor


class IML_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        iml_args = parser.add_argument_group("IML-UAP Attack Arguments")
        
        iml_args.add_argument(
            "--lr",
            type=float,
            default=2e-2,
            metavar="FLOAT",
            help="Learning rate for the adversarial trigger optimization.",
        )

        iml_args.add_argument(
            "--layers",
            type=str,
            nargs="+",
            default=["model.avgpool"],
            metavar="NAMES",
            help="Names of the layers to extract activations from.",
        )

        iml_args.add_argument(
            "--skip_fooled",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that are already fooled by the universal trigger.",
        )

        iml_args.add_argument(
            "--skip_failed",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that the inner attack failed to attack.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr)
        criterion = CosSim(reduce_fn=lambda losses: torch.sum(losses, dim=-1).mean())
        activ_extractor = ActivationExtractor(pert_model.model, **args.layers, capture_output=True) # NOTE: HERE WE DP CAPTURE THE OUTPUT

        def attack_builder(model: torch.nn.Module, epoch: int) -> torchattacks.attack.Attack:
            eps = (pert_model.eps / args.max_epochs) * (epoch + 1)
            # return torchattacks.FFGSM(model, eps=eps, alpha=eps)
            # return torchattacks.PGD(model, eps=eps, alpha=eps / 3, steps=5)
            # return torchattacks.Jitter(model, eps=eps, alpha=eps / 3, steps=5)
            return torchattacks.CW(model, steps=5, lr=eps / 3)

        return IML_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            inner_attack=attack_builder,
            activ_extractor=activ_extractor,
            skip_already_fooled=args.skip_fooled.lower() == "true",
            skip_failed_attacks=args.skip_failed.lower() == "true",
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    IML_Experiment().main()
