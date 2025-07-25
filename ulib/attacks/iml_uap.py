import torch
import torchattacks.attack
from ulib.attack import OptimAttack
from ulib.pert_module import PertModule
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class CosSim(ActivationLoss):
    def __init__(self, aggr_fn=None):
        super().__init__(
            loss_fn=lambda v1, v2: 1.0 - torch.cosine_similarity(v1, v2, dim=-1),
            aggr_fn=aggr_fn,
        )


class L2Diff(ActivationLoss):
    def __init__(self, aggr_fn=None):
        super().__init__(
            loss_fn=lambda v1, v2: torch.sum(torch.square(v1 - v2), dim=-1) / 2,
            aggr_fn=aggr_fn,
        )


class L1Diff(ActivationLoss):
    def __init__(self, aggr_fn=None):
        super().__init__(
            loss_fn=lambda v1, v2: torch.sum(torch.abs(v1 - v2), dim=-1),
            aggr_fn=aggr_fn,
        )


class IML_UAP(OptimAttack):
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: ActivationLoss,
        activ_extractor: ActivationExtractor,
        inner_attack: torchattacks.attack.Attack,
        skip_already_fooled: bool = True,
        skip_failed_attacks: bool = True,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs,
        )

        self.inner_attack = inner_attack
        self.extractor = activ_extractor
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks

        if self.targeted:
            self.inner_attack.set_mode_targeted_by_function(lambda inp, lbl: lbl)

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({f"inner_attack/{k}": v for k, v in inner_attack.__dict__.items()})
        self.logger.register_hparams({"inner_attack/name": inner_attack.__class__.__name__})
        self.logger.register_hparams({"attack/skip_already_fooled": skip_already_fooled})
        self.logger.register_hparams({"attack/skip_failed_attacks": skip_failed_attacks})

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor | None:
        x_batch, y_batch = data

        with self.extractor.capture():
            # record forward pass activations
            y_pred = self.pert_model(x_batch).argmax(dim=1)
            activ = self.extractor.get_activations()

            if self.skip_already_fooled:
                # attack only non-fooled samples
                cor_mask = y_pred == y_batch
                if not cor_mask.any():
                    return None
                x_batch = x_batch[cor_mask]
                y_batch = y_batch[cor_mask]
                activ = {k: v[cor_mask] for k, v in activ.items()}

            # per-sample attack, disable mixed precision
            with self.autocast_context(enabled=False):
                pert = self.pert_model.get_pert()
                x_pert = self.pert_model.clamp_inputs(x_batch + pert)
                x_attk = self.inner_attack.forward(x_pert.detach(), y_batch)

            # record forward pass activations
            y_attk = self.orig_model(x_attk).argmax(dim=1)
            adv_act = self.extractor.get_activations()

            if self.skip_failed_attacks:
                # use only successfully attacked samples
                suc_mask = y_attk != y_batch
                if not suc_mask.any():
                    return None
                activ = {k: v[suc_mask] for k, v in activ.items()}
                adv_act = {k: v[suc_mask] for k, v in adv_act.items()}

        loss = self.criterion(activ, adv_act)
        return loss
