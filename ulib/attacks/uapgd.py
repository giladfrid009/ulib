from typing import Iterable
import torch
import torch.nn as nn
import torchattacks.attack
from ulib.pert_module import PertModule
from ulib.attack import UnivAttack


class ValueScheduler:
    def __init__(self, init_value: float, sched_cls, **kwargs):
        """
        Args:
            init_value (float): Initial value for the scheduler.
            sched_cls (torch.optim.lr_scheduler._LRScheduler): Scheduler class to use.
            **kwargs: Additional keyword arguments to pass to the scheduler.
        """
        # Create a dummy parameter to drive the scheduler.
        self.param = torch.nn.Parameter(torch.tensor([init_value], dtype=torch.float))
        self.optimizer = torch.optim.SGD([self.param], lr=init_value)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = sched_cls(self.optimizer, **kwargs)

        self.optimizer.step()

    def step(self) -> float:
        """
        Perform a step in the scheduler and return the new value.

        Returns:
            float: New value.
        """
        self.scheduler.step()
        return self.get_value()

    def get_value(self) -> float:
        """
        Get the current value of the scheduler.

        Returns:
            float: Current value.
        """
        return self.scheduler.get_last_lr()[0]


class MIPGD(torchattacks.attack.Attack):
    r"""
    MI-PGD as presented in the paper:
    https://ieeexplore.ieee.org/document/9191288

    Distance Measure: Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        beta (float): momentum decay factor.
        gamma (float): weight gain.
        steps (int): number of iterations.

    Shape:
        - images: (N, C, H, W) in [0, 1].
        - labels: (N).
        - output: (N, C, H, W).
    """

    def __init__(self, model: torch.nn.Module, eps=8 / 255, alpha=1.0, beta=0.5, gamma=1e-5, steps=10):
        super().__init__("MIPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = steps
        self.supported_mode = ["default"]

        assert self.eps >= 0, "perturbation norm must be positive"
        assert self.alpha >= 0, "learning rate must be positive"
        assert self.beta >= 0, "momentum decay factor must be positive"
        assert self.gamma >= 0, "weight gain must be positive"

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Move data to device and detach from any computation graph.
        images = images.to(self.device).detach()
        labels = labels.to(self.device).detach()

        # Initialize perturbation (delta) and momentum (v)
        delta = torch.zeros_like(images)
        v = torch.zeros_like(images)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            with torch.enable_grad():
                # Forward and backward pass to compute the gradient
                adv_images = torch.clamp(images + delta, 0, 1)
                adv_images.requires_grad = True
                adv_images.grad = None

                # Compute the current adversarial image and ensure it lies in [0,1]
                outputs = self.get_logits(adv_images)

                # Early stopping
                if (outputs.argmax(dim=1) != labels).all():
                    break

                cost = loss_fn(outputs, labels)
                grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # Update the universal perturbation
            norm_v = torch.linalg.vector_norm(v, ord=2, dim=(1, 2, 3), keepdim=True) + torch.finfo(v.dtype).eps
            delta = delta + self.alpha * (v / norm_v)
            delta = torch.clamp(delta, -self.eps, self.eps)

            # Update the momentum
            v = self.beta * v + grad + self.gamma * adv_images

        # Return the final adversarial images
        adv_images = torch.clamp(images + delta, 0, 1)

        return adv_images


class UAPGD(UnivAttack):
    """
    ## Reference:
        Presented in "Universal Adversarial Attack Via Enhanced Projected Gradient Descent": https://ieeexplore.ieee.org/document/9191288

    Args:
        inner_attack (ulib.attacks.mipgd.MIPGD): Inner attack to use for generating adversarial examples.
            Momentum PGD attack with L2 gradient normalization.
        alpha_sched (ValueScheduler, optional): Scheduler for the learning-rate (alpha) parameter of the inner `MIPGD` attack.
        sched_on_batch (bool): Whether to invoke scheduler update on each batch or each epoch.
    """

    def __init__(
        self,
        pert_model: PertModule,
        inner_attack: MIPGD,
        alpha_sched: ValueScheduler | None = None,
        sched_on_batch: bool = False,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            targeted=False,
            **kwargs,
        )

        self.inner_attack = inner_attack
        self.alpha_sched = alpha_sched
        self.sched_on_batch = sched_on_batch

        self.logger.register_hparams({f"inner_attack/{k}": v for k, v in inner_attack.__dict__.items()})
        self.logger.register_hparams({"inner_attack/name": inner_attack.__class__.__name__})
        self.logger.register_hparams({"attack/sched_on_batch": sched_on_batch})
        if alpha_sched is not None:
            self.logger.register_hparams({f"alpha_sched/{k}": v for k, v in alpha_sched.scheduler.state_dict().items()})
            self.logger.register_hparams({"alpha_sched/name": alpha_sched.scheduler.__class__.__name__})

    def on_batch_start(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int):
        if self.alpha_sched is not None and self.sched_on_batch:
            self.logger.log_scalar("inner_attack/alpha", self.inner_attack.alpha)
            self.inner_attack.alpha = self.alpha_sched.step()

    def on_epoch_start(self, dl_train: Iterable[tuple[torch.Tensor, ...]], epoch_num: int):
        if self.alpha_sched is not None and not self.sched_on_batch:
            self.logger.log_scalar("inner_attack/alpha", self.inner_attack.alpha)
            self.inner_attack.alpha = self.alpha_sched.step()

    @torch.no_grad()
    def process_batch(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> float | None:
        x_batch, y_batch = data

        # attack only correctly classified samples
        y_pred = self.pert_model(x_batch).argmax(dim=1)
        cor_mask = y_pred == y_batch
        if not cor_mask.any():
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
        if not suc_mask.any():
            return None
        x_attk = x_attk[suc_mask]
        x_pert = x_pert[suc_mask]

        delta = torch.sum(x_attk - x_pert, dim=0, keepdim=True)

        pert = self.pert_model.get_pert(clone=False)
        new_pert = self.pert_model.project(pert + delta)
        self.pert_model.set_pert(new_pert)

        return delta.abs().mean().item()
