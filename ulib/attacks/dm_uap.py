import torch
from torch import nn
import torch.nn.functional as F
import copy
import torchattacks.attack
from typing import Callable
import inspect

from ulib.pert_module import PertModule
from ulib.attack import UnivAttack


class PGD_Theta_Apart(torchattacks.attack.Attack):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1.0,
        rho: float = 2,
        data_steps: int = 5,
        theta_steps: int = 10,
        random_start: bool = True,
        eps_for_division: float = 1e-10,
        smooth_rate: float = 0,
    ):
        """
        Code Reference: https://github.com/yechao-zhang/DM-UAP/blob/main/PGDL2.py

        Args:
            model (nn.Module): The raw model to attack.
            eps (float): The maximum L2 norm of the adversarial perturbation.
            rho (float): The maximum L2 norm of the model parameter perturbation.
            data_steps (int): Number of data optimization steps in the inner PGD attack (Kd in paper).
            theta_steps (int): Number of steps of model param optimization in the inner PGD attack (Km in paper).
            random_start (bool): Whether to start the inner PGD attack with a random initialization.
            eps_for_division (float): A small constant added to the denominator for numerical stability when normalizing gradients.
            smooth_rate (float): The smoothing rate used in the GLS loss. Could go either positive or negative.
        """
        super().__init__("PGD_Theta_Apart", model)

        self.eps = eps
        self.alpha = eps * 0.125
        self.data_steps = data_steps
        self.theta_steps = theta_steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.rho = rho
        self.smooth_rate = smooth_rate

        self.supported_mode = ["default"]

    def loss_gls(self, logits: torch.Tensor, labels: torch.Tensor, smooth_rate: float) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): model prediction logits before the soft-max, with size [batch_size, classes]
            labels (torch.Tensor): the (noisy) labels for evaluation, with size [batch_size]
            smooth_rate (float): could go either positive or negative, smooth_rate candidates
                we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
        """

        confidence = 1.0 - smooth_rate
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smooth_rate * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return torch.sum(loss) / num_batch

    def ce_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(output, target)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.rho == 0 and self.alpha == 0:
            return adv_images

        for i in range(self.theta_steps):
            step_rho = self.rho / self.theta_steps

            self.model.train()
            self.model.zero_grad()
            for p in self.model.parameters():
                p.requires_grad = True

            outputs = self.get_logits(adv_images)
            loss = self.loss_gls(outputs, labels, self.smooth_rate)
            loss.backward()

            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in self.model.parameters() if p.grad is not None]), p=2)
            scale = -step_rho / (grad_norm + 1e-12)
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                p.data.add_(p.grad * scale)

            self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        if self.random_start and self.data_steps:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()
        
        for _ in range(self.data_steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost = -self.ce_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        adv_images.requires_grad = False
        return adv_images.detach().clone()


class DM_UAP(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        inner_attack: PGD_Theta_Apart | Callable[[torch.nn.Module, int], PGD_Theta_Apart],
        beta: float = 9.0,
        **kwargs,
    ):
        """
        ## Reference:
            Presented in "Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization": https://arxiv.org/pdf/2503.12793

        Code Reference: https://github.com/yechao-zhang/DM-UAP

        Args:
            pert_model (PertModule): A perturbation model used to generate adversarial examples.
            optimizer (torch.optim.Optimizer): Optimizer for training the perturbation model.
            inner_attack (PGD_Theta_Apart | Callable[[torch.nn.Module, int], PGD_Theta_Apart]): The inner attack used to generate adversarial examples and update model parameters.
                Can be either a fixed instance of PGD_Theta_Apart or a callable that takes the current model and epoch number as input and returns a PGD_Theta_Apart instance (for curriculum scheduling).
            beta (float): Loss clamping value.
        """
        super().__init__(
            pert_model=pert_model,
            targeted=False,
            **kwargs,
        )

        if isinstance(inner_attack, PGD_Theta_Apart):
            self.attack_builder_func = None
            self.inner_attack = inner_attack
        elif callable(inner_attack):
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(self.orig_model, 0)
        else:
            raise ValueError("inner_attack must be either an instance of PGD_Theta_Apart or a callable that returns a PGD_Theta_Apart instance.")

        self.beta = beta
        self.optimizer = optimizer
        self.inner_attack = self.make_attack(0)
        self.orig_state_dict = copy.deepcopy(self.orig_model.state_dict())

        self.metric_tracker.report_hparams(
            "inner_attack",
            self.inner_attack.__dict__,
            name=type(self.inner_attack).__name__,
        )

        self.metric_tracker.report_hparams(
            "optim",
            self.optimizer.param_groups[0],
            name=type(self.optimizer).__name__,
        )

        self.metric_tracker.report_hparams(
            "attack",
            inner_attack=type(self.inner_attack).__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            beta=self.beta,
        )

    def clamped_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = torch.mean(torch.clamp(loss_fn(output, target), max=self.beta))
        return loss

    def make_attack(self, epoch_num: int) -> PGD_Theta_Apart:
        if self.attack_builder_func is None:
            inner_attack = self.inner_attack
        else:
            inner_attack = self.attack_builder_func(self.orig_model, epoch_num)
        return inner_attack

    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:

        if batch_num == 0 and epoch_num > 0:  # on epoch start
            self.make_attack(epoch_num)

        x_batch, _ = data
        model = self.orig_model

        with torch.no_grad():
            outputs_ori = model(x_batch)
            _, target_label = torch.max(outputs_ori, dim=1)

        self.optimizer.zero_grad()

        # perturbs x_batch and also modifies the model parameters
        x_val = self.inner_attack(x_batch, target_label)
        pert = self.pert_model.get_pert(clone=False)
        outputs = model(x_val + pert)

        loss = -self.clamped_loss(outputs, target_label)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pert = self.pert_model.project(pert)
            self.pert_model.set_pert(pert)

        model.load_state_dict(self.orig_state_dict)
        model.eval()

        return loss.item()
