import torch
import torch.nn as nn
import torchattacks
import torchattacks.attack
from ulib.attack import OptimAttack
from ulib.pert_module import PertModule
from typing import Callable
import inspect


class AELoss(torch.nn.Module):
    def __init__(self, cr_lambda: float):
        super().__init__()
        assert cr_lambda >= 0, "Regularization strength must be non-negative"
        self.gamma = cr_lambda

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logits_loss(logits, y) + self.gamma * self.consistency_loss(logits)

    def logits_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -torch.gather(logits, 1, y.view(-1, 1)).mean()

    def consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # TODO: FIX, produces infinite values
        return 0  # type: ignore
        logits = logits.view(logits.size(0), -1)
        logits = nn.functional.normalize(logits, p=2, dim=1)
        det = torch.det(logits @ logits.T)
        return torch.log(det).mean()


class AE_MIFGSM(torchattacks.attack.Attack):
    """
    MI-FGSM variant presented in https://ojs.aaai.org/index.php/AAAI/article/view/20023
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 20 / 255,
        steps: int = 10,
        decay: float = 1.0,
    ):
        """
        Args:
            model (nn.Module): The model to attack.
            eps (float): The maximum L2 norm of the perturbation.
            steps (int): The number of attack iterations.
            decay (float): The decay factor for the momentum term.
        """
        super().__init__("AE_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.supported_mode = ["default"]

    @property
    def alpha(self) -> float:
        """Step size of the attack, calculated as eps / steps."""
        return self.eps / self.steps

    def set_criterion(self, criterion: torch.nn.Module):
        self.criterion = criterion

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            cost = self.criterion(outputs, labels)
            print(f"Cost: {cost.mean().item()}")

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1) + torch.finfo(grad.dtype).eps
            grad_norm = grad_norm.reshape(-1, 1, 1, 1)

            momentum = self.decay * momentum - grad / grad_norm
            momentum_norm = torch.norm(momentum.view(momentum.size(0), -1), p=2, dim=1) + torch.finfo(momentum.dtype).eps
            momentum_norm = momentum_norm.reshape(-1, 1, 1, 1)

            adv_images = adv_images - self.alpha * momentum / momentum_norm
            adv_images = torch.clamp(adv_images, 0, 1)

            # L2 projection
            delta = adv_images - images
            delta = torch.renorm(delta, p=2, dim=0, maxnorm=self.eps)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class AE_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Learning Universal Adversarial Perturbation by Adversarial Example": https://ojs.aaai.org/index.php/AAAI/article/view/20023

    Args:
        inner_attack (ulib.attacks.ae_uap.AE_MIFGSM): Inner attack to use for generating adversarial examples.
        data_free (bool): Whether to use a data-free version of the attack.
        cr_lambda (float): Regularization strength for the consistency loss term, in the `AELoss` class.
        noise_scale (float): Uniform noise scale to use for the data-free version of the attack.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        inner_attack: AE_MIFGSM | Callable[[torch.nn.Module, int], AE_MIFGSM],
        data_free: bool = False,
        cr_lambda: float = 0.5,
        noise_scale: float = 16 / 255,
        **kwargs,
    ):
        if not isinstance(inner_attack, AE_MIFGSM):
            raise ValueError("inner_attack must be an instance of AE_MIFGSM")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=AELoss(cr_lambda=cr_lambda),
            targeted=False,
            **kwargs,
        )

        if isinstance(inner_attack, AE_MIFGSM):
            self.attack_builder_func = None
            self.inner_attack = inner_attack
        else:
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(self.orig_model, 0)

        self.inner_attack = self.make_attack(0)
        self.data_free = data_free
        self.noise_scale = noise_scale

        self.metric_tracker.report_hparams(
            "inner_attack",
            inner_attack.__dict__,
            name=type(inner_attack).__name__,
        )

        self.metric_tracker.report_hparams(
            "attack",
            inner_attack=type(self.inner_attack).__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            cr_lambda=cr_lambda,
            data_free=data_free,
            noise_scale=noise_scale,
        )

    def make_attack(self, epoch_num: int) -> AE_MIFGSM:
        if self.attack_builder_func is None:
            inner_attack = self.inner_attack
        else:
            inner_attack = self.attack_builder_func(self.orig_model, epoch_num)
        if self.targeted:
            inner_attack.set_mode_targeted_by_function(lambda inp, lbl: lbl)

        inner_attack.set_criterion(self.criterion)
        return inner_attack

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor:

        # schedule every step
        self.inner_attack = self.make_attack(step_num)

        x_batch, y_batch = data

        if self.data_free:
            x_batch = torch.full_like(x_batch, fill_value=0.5)
            noise = (2 * torch.rand_like(x_batch) - 1) * self.noise_scale
            x_batch = torch.clamp(x_batch + noise, min=0, max=1)
            # NOTE: we use the real y_batch, its equivalent to using random labels
            # otherwise, we can simply randomize the labels as well

        with self.autocast_context(enabled=False):
            x_attk = self.inner_attack(x_batch, y_batch)
        logits = self.pert_model(x_attk)
        loss = -self.criterion(logits, y_batch)
        return loss
