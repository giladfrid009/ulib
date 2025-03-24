from typing import Iterable
import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        decisions = logits.argmax(dim=1)
        onehot_preds = torch.nn.functional.one_hot(decisions, num_classes=logits.size(1))
        onehot_y = torch.nn.functional.one_hot(y, num_classes=logits.size(1))
        return 1.0 - (onehot_preds != onehot_y).float().mean()


class D_BADGE(OptimAttack):
    """
    Reference:
        Presented in "D-BADGE: Decision-Based Adversarial Batch Attack With Directional Gradient Estimation": https://ieeexplore.ieee.org/document/10542123
        Code inspired from original peper repo: https://github.com/AIRLABkhu/D-BADGE
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        delta: float = 0.01,
        delta_decay: float = 0.9,
        gamma: float = 0.001,
        **kwargs,
    ):
        if delta <= 0:
            raise ValueError("delta must be > 0")

        if delta_decay < 0 or delta_decay > 1:
            raise ValueError("delta_decay must be in [0, 1]")

        if gamma <= 0:
            raise ValueError("gamma must be > 0")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=HammingLoss(),
            targeted=False,
            **kwargs,
        )

        self.delta = delta
        self.delta_decay = delta_decay
        self.gamma = gamma

        self.logger.register_hparams({"attack/delta": delta})
        self.logger.register_hparams({"attack/delta_decay": delta_decay})
        self.logger.register_hparams({"attack/gamma": gamma})

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        x_batch, y_batch = data
        preds = self.pert_model(x_batch)
        loss = self.criterion(preds, y_batch)
        return loss

    def on_epoch_start(self, dl_train: Iterable[tuple[torch.Tensor, ...]], epoch_num: int):
        if epoch_num > 0:
            # Decay delta
            self.delta = self.delta * self.delta_decay

    @torch.no_grad()
    def process_batch(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> float | None:
        self.optimizer.zero_grad()

        pert = self.pert_model.get_pert(clone=False)
        rnd_signs = (torch.rand_like(pert) > 0.5).float() * 2.0 - 1.0
        vec = self.delta * rnd_signs

        with self.autocast_context():
            self.pert_model.set_pert(pert - vec)
            loss_neg = self.compute_loss(data, batch_num, epoch_num)

            self.pert_model.set_pert(pert + vec)
            loss_pos = self.compute_loss(data, batch_num, epoch_num)

        # Restore perturbation
        self.pert_model.set_pert(pert)

        # Compute loss
        loss = loss_neg - loss_pos

        # Scale loss if needed
        if self.grad_scaler is not None:
            loss = self.grad_scaler.scale(loss)

        # Compute gradient
        grad = loss / (self.gamma * vec)
        pert.grad = grad

        # Update model
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        # Update learning rate scheduler if enabled.
        if self.scheduler is not None and (self.sched_on_batch or batch_num == 0):
            self.logger.log_scalar("lr", self.scheduler.get_last_lr()[0])
            self.scheduler.step()

        return pert.grad.norm().item()
