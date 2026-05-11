import torch
from typing import Callable
import inspect

from torchattacks.attack import Attack as TorchAttack
from ulib.pert_module import PertModule
from ulib.attack import UnivAttack


class UAP(UnivAttack):
    """
    ## Reference:
        - When `targeted=False`: Presented in "Universal adversarial perturbations": https://arxiv.org/abs/1610.08401
        - When `targeted=True`: Presented in "Simple iterative method for generating targeted universal adversarial perturbations": https://arxiv.org/pdf/1911.06502

    Args:
        inner_attack (torchattacks.attack.Attack): Inner attack to use for generating adversarial examples.
    """

    def __init__(
        self,
        pert_model: PertModule,
        inner_attack: TorchAttack | Callable[[torch.nn.Module, int], TorchAttack],
        targeted=False,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            targeted=targeted,
            **kwargs,
        )

        if isinstance(inner_attack, TorchAttack):
            self.attack_builder_func = None
            self.inner_attack = inner_attack
        else:
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(self.orig_model, 0)

        self.inner_attack = self.make_attack(0)

        self.metric_tracker.report_hparams(
            "inner_attack",
            self.inner_attack.__dict__,
            name=type(self.inner_attack).__name__,
        )
        self.metric_tracker.report_hparams(
            "attack",
            inner_attack=type(self.inner_attack).__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
        )

    def make_attack(self, epoch_num: int) -> TorchAttack:
        if self.attack_builder_func is None:
            inner_attack = self.inner_attack
        else:
            inner_attack = self.attack_builder_func(self.orig_model, epoch_num)
        if self.targeted:
            inner_attack.set_mode_targeted_by_function(lambda inp, lbl: lbl)
        return inner_attack

    @torch.no_grad()
    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:
        if batch_num == 0:  # on epoch start
            self.inner_attack = self.make_attack(epoch_num)

        x_batch, y_batch = data

        # attack only non-fooled samples
        y_pred = self.pert_model(x_batch).argmax(dim=1)
        cor_mask = y_pred == y_batch

        if self.targeted:
            cor_mask = ~cor_mask
        
        not_fooled_ratio = cor_mask.float().mean().item()
        self.metric_tracker.report_scalar("UAP/not_fooled_ratio", not_fooled_ratio, step_num)

        if not cor_mask.any():
            self.metric_tracker.report_scalar("UAP/effective_batch_ratio", 0.0, step_num)
            return None
  
        x_batch = x_batch[cor_mask]
        y_batch = y_batch[cor_mask]

        # per-sample attack
        pert = self.pert_model.get_pert(clone=False)
        x_pert = self.pert_model.clamp_inputs(pert + x_batch)
        x_attk = self.inner_attack.forward(x_pert.detach(), y_batch)

        # use only successfully attacked samples
        y_attk = self.orig_model(x_attk).argmax(dim=1)
        suc_mask = y_attk != y_batch
        
        if self.targeted:
            suc_mask = ~suc_mask
                    
        sample_asr = suc_mask.float().mean().item()
        self.metric_tracker.report_scalar("UAP/sample_attack_success_ratio", sample_asr, step_num)

        if not suc_mask.any():
            self.metric_tracker.report_scalar("UAP/effective_batch_ratio", 0.0, step_num)
            return None
        
        x_attk = x_attk[suc_mask]
        x_pert = x_pert[suc_mask]

        num_remaining = x_pert.size(0)
        num_initial = data[0].size(0)
        batch_ratio = num_remaining / num_initial
        self.metric_tracker.report_scalar("UAP/effective_batch_ratio", batch_ratio, step_num)

        delta = torch.sum(x_attk - x_pert, dim=0, keepdim=True)

        pert = self.pert_model.get_pert(clone=False)
        new_pert = self.pert_model.project(pert + delta)
        self.pert_model.set_pert(new_pert)

        return delta.abs().mean().item()
