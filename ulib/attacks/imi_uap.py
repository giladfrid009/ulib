import torch
import torchattacks.attack
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class IMI_UAP(OptimAttack):
    """
    Args:
        inner_attack (torchattacks.attack.Attack): Inner attack to be used for generating
            adversarial examples for each individual sample.
        skip_already_fooled (bool): Skip samples that are already fooled by the current perturbation.
        skip_failed_attacks (bool): Skip samples for which the inner attack fails to generate an adversarial example.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        inner_attack: torchattacks.attack.Attack,
        skip_already_fooled: bool = True,
        skip_failed_attacks: bool = True,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            targeted=False,
            **kwargs,
        )

        self.inner_attack = inner_attack
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks

        self.logger.register_hparams({f"inner_attack/{k}": v for k, v in inner_attack.__dict__.items()})
        self.logger.register_hparams({"inner_attack/name": inner_attack.__class__.__name__})
        self.logger.register_hparams({"attack/skip_already_fooled": skip_already_fooled})
        self.logger.register_hparams({"attack/skip_failed_attacks": skip_failed_attacks})

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor | None:
        x_batch, y_batch = data

        if self.skip_already_fooled:
            # attack only non-fooled samples
            with torch.no_grad():
                y_pred = self.pert_model(x_batch).argmax(dim=1)
                mask = y_pred == y_batch
                if not mask.any():
                    return None
                x_batch = x_batch[mask]
                y_batch = y_batch[mask]

        # per-sample attack
        with self.autocast_context(enabled=False):
            pert = self.pert_model.get_pert(clone=False)
            x_pert = self.pert_model.clamp_inputs(pert + x_batch)
            x_attk = self.inner_attack.forward(x_pert.detach(), y_batch)

        if self.skip_failed_attacks:
            # use only successfully attacked samples
            y_attk = self.orig_model(x_attk).argmax(dim=1)
            mask = y_attk != y_batch
            if not mask.any():
                return None
            x_attk = x_attk[mask]
            x_pert = x_pert[mask]

        # Minimize distance between x_attk and x_pert
        loss = self.criterion(x_attk, x_pert)
        return loss
