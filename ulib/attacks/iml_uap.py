from typing import Callable
import torch
from torchattacks.attack import Attack as TorchAttack
import inspect
from ulib.attack import OptimAttack
from ulib.pert_module import PertModule
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class CosSim(ActivationLoss):
    def __init__(self, reduction="sum-mean"):
        super().__init__(
            loss_fn=lambda v1, v2: 1.0 - torch.cosine_similarity(v1, v2, dim=-1),
            reduction=reduction,
        )


class L2Diff(ActivationLoss):
    def __init__(self, reduction="sum-mean"):
        super().__init__(
            loss_fn=lambda v1, v2: torch.sum(torch.square(v1 - v2), dim=-1) / 2,
            reduction=reduction,
        )


class L1Diff(ActivationLoss):
    def __init__(self, reduction="sum-mean"):
        super().__init__(
            loss_fn=lambda v1, v2: torch.sum(torch.abs(v1 - v2), dim=-1),
            reduction=reduction,
        )


class IML_UAP(OptimAttack):
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: ActivationLoss,
        activ_extractor: ActivationExtractor,
        inner_attack: TorchAttack | Callable[[torch.nn.Module, int], TorchAttack],
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

        if isinstance(inner_attack, TorchAttack):
            self.attack_builder_func = None
            self.inner_attack = inner_attack
        else:
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(self.orig_model, 0)

        self.inner_attack = self.make_attack(0)
        self.extractor = activ_extractor
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks

        self.metric_logger.report_hparams("activ_extractor", activ_extractor.get_hparams())
        self.metric_logger.report_hparams(
            "inner_attack",
            self.inner_attack.__dict__,
            name=self.inner_attack.__class__.__name__,
        )
        self.metric_logger.report_hparams(
            "attack",
            inner_attack=self.inner_attack.__class__.__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            skip_already_fooled=skip_already_fooled,
            skip_failed_attacks=skip_failed_attacks,
        )

    def make_attack(self, epoch_num: int) -> TorchAttack:
        if self.attack_builder_func is None:
            inner_attack = self.inner_attack
        else:
            inner_attack = self.attack_builder_func(self.orig_model, epoch_num)
        if self.targeted:
            inner_attack.set_mode_targeted_by_function(lambda inp, lbl: lbl)
        return inner_attack

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor | None:
        if batch_num == 0:  # on epoch start
            self.inner_attack = self.make_attack(epoch_num)

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
