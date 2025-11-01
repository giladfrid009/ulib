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
        pass

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_logger):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=2e-2)
        criterion = CosSim(reduce_fn=lambda losses: torch.prod(losses, dim=-1).mean())
        activ_extractor = ActivationExtractor(pert_model.model, "model.avgpool", exact_match=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=5e-5)

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
            skip_already_fooled=True,
            skip_failed_attacks=True,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_logger=metric_logger,
        )


if __name__ == "__main__":
    IML_Experiment().main()
