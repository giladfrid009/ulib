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
    ## Reference:
        Presented in "D-BADGE: Decision-Based Adversarial Batch Attack With Directional Gradient Estimation": https://ieeexplore.ieee.org/document/10542123
        Code inspired from original paper repo: https://github.com/AIRLABkhu/D-BADGE

    Args:
        delta (float): L-inf radius of the random noise at which the gradient is estimated.
        delta_decay (float): Factor by which to decay delta each epoch.
        gamma (float): Scaling factor for the gradient.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        delta: float = 0.01,
        delta_decay: float = 0.9,
        gamma: float = 1000,
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

        self.metric_logger.report_hparams(
            "attack",
            delta=delta,
            delta_decay=delta_decay,
            gamma=gamma,
        )

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor:
        x_batch, y_batch = data
        preds = self.pert_model(x_batch)
        loss = self.criterion(preds, y_batch)
        return loss

    @torch.no_grad()
    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:
        if batch_num == 0 and epoch_num > 0:
            # Decay delta
            self.delta = self.delta * self.delta_decay

        self.optimizer.zero_grad()

        pert = self.pert_model.get_pert(clone=False)
        rnd_signs = (torch.rand_like(pert) > 0.5).float() * 2.0 - 1.0
        vec = self.delta * rnd_signs

        with self.autocast_context():
            self.pert_model.set_pert(pert - vec)
            loss_neg = self.compute_loss(data, batch_num, epoch_num, step_num)

            self.pert_model.set_pert(pert + vec)
            loss_pos = self.compute_loss(data, batch_num, epoch_num, step_num)

        # Restore perturbation
        self.pert_model.set_pert(pert)

        # Compute loss
        loss = loss_neg - loss_pos

        # Scale loss if needed
        if self.grad_scaler is not None:
            loss = self.grad_scaler.scale(loss)

        # Compute gradient
        grad = self.gamma * loss / vec
        pert.grad = grad

        # Update model
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        # Update learning rate scheduler if enabled.
        if self.scheduler is not None and (self.sched_on_batch or batch_num == 0):
            self.metric_logger.report_scalar("lr", self.scheduler.get_last_lr()[0], step_num)
            self.scheduler.step()

        return pert.grad.norm().item()
