from typing import Iterable
import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class SD_UAP(OptimAttack):
    """
    Reference:
        Presented in "CRAFTING DATA-FREE UNIVERSAL ADVERSARIES WITH DILATE LOSS": https://openreview.net/attachment?id=HJxVC1SYwr&name=original_pdf
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        data_dependant: bool = True,
        progress_every: int = 2,
        **kwargs,
    ):
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        
        criterion = ActivationLoss(loss_fn=lambda v: -torch.log(torch.sum(torch.square(v) / 2, dim=1) + torch.finfo(v.dtype).eps))

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            targeted=False,
            **kwargs,
        )

        self.data_dependant = data_dependant
        self.progress_every = progress_every
        self.layers = activ_extractor.layer_names
        
        self.cur_idx = 0
        self.cur_extractor = activ_extractor
        self.update_extractor(idx=0)

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({"attack/layers": str(self.layers)})
        self.logger.register_hparams({"attack/data_dependant": data_dependant})
        self.logger.register_hparams({"attack/progress_every": progress_every})

    def update_extractor(self, idx: int | None = None): 
        new_idx = idx if idx is not None else self.cur_idx + 1
        self.cur_idx = min(new_idx, len(self.layers) - 1)
        self.cur_extractor = ActivationExtractor(self.orig_model, *self.layers[: self.cur_idx + 1])

    def on_epoch_start(self, dl_train: Iterable[tuple[torch.Tensor, ...]], epoch_num: int):
        if epoch_num > 0 and epoch_num % self.progress_every == 0:
            # progress to next layer
            self.update_extractor()

            # divide pert by two
            pert = self.pert_model.get_pert(clone=False)
            with torch.no_grad():
                pert.divide_(2.0)
        
        self.logger.log_scalar("attack/layer_index", self.cur_idx)

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        x_batch, _ = data
        
        if not self.data_dependant:
            # Zero out input if not data dependant
            x_batch = torch.zeros_like(x_batch)

        with self.cur_extractor.capture():
            self.pert_model(x_batch)
            activ = self.cur_extractor.get_activations()

        loss = self.criterion(activ)
        return loss

    def on_batch_end(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int):
        # Set perturbation to have exactly eps norm
        with torch.no_grad():
            pert = self.pert_model.get_pert(clone=False)
            pert_norm = torch.norm(pert.flatten(), p=self.pert_model.norm)
            pert.divide_(pert_norm / self.pert_model.eps)
