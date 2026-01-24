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


class UAP_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        uap_args = parser.add_argument_group("UAP Attack Arguments")

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        
        def attack_builder(model: torch.nn.Module, epoch: int) -> torchattacks.attack.Attack:
            eps = (pert_model.eps / args.max_epochs) * (epoch + 1)
            # return torchattacks.FFGSM(model, eps=eps, alpha=eps)
            # return torchattacks.PGD(model, eps=eps, alpha=eps / 3, steps=5)
            # return torchattacks.Jitter(model, eps=eps, alpha=eps / 3, steps=5)
            return torchattacks.CW(model, steps=5, lr=eps / 3)
        
        return UAP(
            pert_model=pert_model,
            inner_attack=attack_builder,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    UAP_Experiment().main()
