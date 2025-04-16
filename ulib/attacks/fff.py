import torch
from typing import Iterable
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class FFF(OptimAttack):
    """
    ## Reference:
        Presented in "Fast Feature Fool: A data independent approach to universal adversarial perturbations": https://arxiv.org/pdf/1707.05572

    Args:
        divide_every (float): Divide the perturbation by 2 every `divide_every` epochs.
            Set to 0 to disable this feature.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        divide_every: float = 0,
        **kwargs,
    ):
        if divide_every < 0:
            raise ValueError("`divide_every` must be >= 0")

        criterion = ActivationLoss(loss_fn=lambda v: -torch.log(torch.mean(torch.abs(v), dim=1) + torch.finfo(v.dtype).eps))

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            targeted=False,
            **kwargs,
        )

        self.divide_every = divide_every
        self.extractor = ActivationExtractor(pert_model.model, torch.nn.Conv2d)
        self.logger.register_hparams(self.extractor.get_hparams())
        self.logger.register_hparams({"attack/divide_every": divide_every})

    def on_epoch_start(self, dl_train: Iterable[tuple[torch.Tensor, ...]], epoch_num: int):
        if epoch_num > 0 and self.divide_every > 0 and epoch_num % self.divide_every == 0:
            with torch.no_grad():
                pert = self.pert_model.get_pert(clone=False)
                pert.divide_(2.0)

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        input = torch.zeros(self.pert_model.shape)
        with self.extractor.capture():
            self.pert_model(input)
            activ = self.extractor.get_activations()
        loss = self.criterion(activ)
        return loss
