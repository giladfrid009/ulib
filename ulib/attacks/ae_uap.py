import torch
import torch.nn as nn
import torchattacks
import torchattacks.attack
from ulib.attack import OptimAttack
from ulib.pert_module import PertModule
import torch


class AELoss(torch.nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        assert gamma >= 0, "Regularization strength must be non-negative"
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logits_loss(logits, y) + self.gamma * self.consistency_loss(logits)

    def logits_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -torch.gather(logits, 1, y.view(-1, 1)).mean()

    def consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.view(logits.size(0), -1)
        norm = torch.norm(logits, p=2, dim=1, keepdim=True)
        logits = logits / (norm + torch.finfo(norm.dtype).eps)
        det = torch.sum(logits * logits, dim=1)
        return torch.log(det).mean()


class AE_MIFGSM(torchattacks.attack.Attack):
    """
    MI-FGSM variant presented in https://ojs.aaai.org/index.php/AAAI/article/view/20023
    """
    def __init__(
        self,
        model: nn.Module,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=1.0,
    ):
        super().__init__("AE_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default"]

    def set_criterion(self, criterion: torch.nn.Module):
        self.criterion = criterion

    def forward(self, images, labels):
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

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1) + torch.finfo(grad.dtype).eps
            grad_norm = grad_norm.reshape(-1, 1, 1, 1)

            momentum = self.decay * momentum - grad / grad_norm
            momentum_norm = torch.norm(momentum.view(momentum.size(0), -1), p=2, dim=1) + torch.finfo(momentum.dtype).eps
            momentum_norm = momentum_norm.reshape(-1, 1, 1, 1)

            adv_images = adv_images - self.alpha * momentum / momentum_norm
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


# TODO: IMPLEMENT DATA-FREE AND DATA-DEPENDENT VERSIONS
# TODO: IMPLEMENT TARGETED VARSION
class AE_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Learning Universal Adversarial Perturbation by Adversarial Example": https://ojs.aaai.org/index.php/AAAI/article/view/20023
            
    Args:
        inner_attack (ulib.attacks.ae_uap.AE_MIFGSM): Inner attack to use for generating adversarial examples.
        gamma (float): Regularization strength for the consistency loss term, in the `AELoss` class.
    """
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        inner_attack: AE_MIFGSM,
        gamma: float = 0.5,
        **kwargs,
    ):
        if not isinstance(inner_attack, AE_MIFGSM):
            raise ValueError("inner_attack must be an instance of AE_MIFGSM")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=AELoss(gamma=gamma),
            targeted=False,
            **kwargs,
        )        

        self.inner_attack = inner_attack
        self.inner_attack.set_criterion(self.criterion)
        
        self.logger.register_hparams({f"inner_attack/{k}": v for k, v in inner_attack.__dict__.items()})
        self.logger.register_hparams({"inner_attack/name": inner_attack.__class__.__name__})

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        x_batch, y_batch = data
        with self.autocast_context(enabled=False):
            x_attk = self.inner_attack.forward(x_batch, y_batch)
        logits = self.pert_model(x_attk)
        loss = -self.criterion(logits, y_batch)
        return loss
