import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class USGD(OptimAttack):
    """
    Universal gradient-based attack that uses an optimizer to update the perturbation.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        skip_already_fooled: bool = True,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs,
        )

        self.skip_already_fooled = skip_already_fooled
        self.logger.register_hparams({"attack/skip_already_fooled": skip_already_fooled})

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor | None:
        x_batch, y_batch = data
        preds = self.pert_model(x_batch)

        if self.skip_already_fooled:
            mask = preds.argmax(dim=1) == y_batch
            if self.targeted:
                mask = ~mask

            if not mask.any():
                return None

            x_batch = x_batch[mask]
            y_batch = y_batch[mask]
            preds = preds[mask]

        loss = self.criterion(preds, y_batch)
        if self.targeted:
            loss = -loss
        return loss
