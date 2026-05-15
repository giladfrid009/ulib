import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from typing import Optional
import copy

from ulib.utils.torch import extract_device
from ulib.pert_module import PertModule
from ulib.attack import UnivAttack


# TODO: clean up code.
class PGD_Theta_Apart:
    """
    Code Reference: https://github.com/yechao-zhang/DM-UAP/blob/main/PGDL2.py
    """

    def __init__(
        self,
        eps: float = 1.0,
        rho: float = 2,
        data_steps: int = 5,
        theta_steps: int = 10,
        random_start: bool = True,
        eps_for_division: float = 1e-10,
        smooth_rate: float = 0,
    ):
        self.eps = eps
        self.alpha = eps * 0.125
        self.x_steps = data_steps
        self.theta_steps = theta_steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.rho = rho
        self.smooth_rate = smooth_rate

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
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        uap: Optional[torch.Tensor] = None,
    ) -> tuple[nn.Module, torch.Tensor]:

        device = extract_device(model)
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        adv_images = images.clone().detach()
        batch_size = len(images)
        if self.rho == 0 and self.alpha == 0:
            return model, adv_images

        for i in range(self.theta_steps):
            step_rho = self.rho / self.theta_steps

            model.train()
            model.zero_grad()
            for p in model.parameters():
                p.requires_grad = True

            outputs = model(adv_images)
            loss = self.loss_gls(outputs, labels, self.smooth_rate)
            loss.backward()

            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]), p=2)
            scale = -step_rho / (grad_norm + 1e-12)
            for p in model.parameters():
                if p.grad is None:
                    continue
                p.data.add_(p.grad * scale)

            model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        if self.random_start and self.x_steps:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()
        for _ in range(self.x_steps):
            adv_images.requires_grad = True
            if uap is not None:
                outputs = model(adv_images + uap)
            else:
                outputs = model(adv_images)
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
        # model.load_state_dict(after_state_dict)

        return model, adv_images.detach().clone()


class DM_UAP(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        total_epochs: int = 20,
        beta: float = 9.0,
        theta_neighborhood: float = 1.0,
        data_neighborhood: float = 32,
        data_steps: int = 10,
        theta_steps: int = 10,
        smooth_rate: float = 0.2,
        **kwargs,
    ):
        """
        ## Reference:
            Presented in "Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization": https://arxiv.org/pdf/2503.12793

        Code Reference: https://github.com/yechao-zhang/DM-UAP

        Args:
            total_epochs (int): Number of training epochs (T in paper).
            beta (float): Loss clamping value.
            theta_neighborhood (float): Model maximum neighborhood size (Rho in paper).
            data_neighborhood (float): Data maximum neighborhood size (r in paper).
            data_steps (int): Number of data optimization steps in the inner PGD attack (Kd in paper).
            theta_steps (int): Number of steps of model param optimization in the inner PGD attack (Km in paper).
            smooth_rate (float):
        """
        super().__init__(pert_model=pert_model, **kwargs)

        self.optimizer=optimizer
        self.total_epochs = total_epochs
        self.beta = beta
        self.theta_neighborhood = theta_neighborhood
        self.data_neighborhood = data_neighborhood
        self.data_steps = data_steps
        self.theta_steps = theta_steps
        self.smooth_rate = smooth_rate

        self.orig_state_dict = copy.deepcopy(self.orig_model.state_dict())

        self.inner_attack = self.make_attack(0)

        self.metric_tracker.report_hparams(
            "attack",
            total_epochs=self.total_epochs,
            beta=self.beta,
            rho=self.theta_neighborhood,
            aa=self.data_neighborhood,
            data_steps=self.data_steps,
            theta_steps=self.theta_steps,
            smooth_rate=self.smooth_rate,
        )

    def clamped_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = torch.mean(torch.clamp(loss_fn(output, target), max=self.beta))  # type: ignore
        return loss

    def make_attack(self, epoch_num: int) -> PGD_Theta_Apart:
        rho_step = self.theta_neighborhood / self.total_epochs
        eps_step = self.data_neighborhood / self.total_epochs
        rho_current = rho_step * (epoch_num + 1)
        eps_current = eps_step * (epoch_num + 1)

        self.inner_attack = PGD_Theta_Apart(
            eps=eps_current,
            rho=rho_current,
            data_steps=self.data_steps,
            theta_steps=self.theta_steps,
            random_start=False,
            eps_for_division=1e-10,
            smooth_rate=self.smooth_rate,
        )

        return self.inner_attack

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
        model, x_val = self.inner_attack.forward(model, x_batch, target_label)

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
