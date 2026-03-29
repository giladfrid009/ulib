import torch
import torch.nn as nn

from sample_attacks.utils import clamp, img_max, img_min


class Attack(object):
    """
    Base class for all attacks.
    """

    def __init__(
        self,
        attack: str,
        model: nn.Module,
        epsilon: float,
        targeted: bool,
        random_start: bool,
        norm: str,
        loss: str,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Initialize the hyperparameters

        Arguments:
            attack (str): the name of attack.
            model (torch.nn.Module): the surrogate model instance for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            random_start (bool): whether using random initialization for delta.
            norm (str): the norm of perturbation, l2/linfty.
            loss (str): the loss function.
            device (torch.device): the device for data. If it is None, the device would be same as model.
        """
        if norm not in ["l2", "linfty"]:
            raise Exception("Unsupported norm {}".format(norm))
        self.attack = attack
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.device = next(self.model.parameters()).device if device is None else device
        self.loss = self.loss_function(loss)

    def forward(self, data: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargeted
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def get_logits(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The inference stage, which should be overridden when the attack needs to
        change the models or the input transformation.
        """
        return self.model(x)

    def get_loss(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        The loss calculation.
        """
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    def get_grad(self, loss: torch.Tensor, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The gradient calculation.
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad: torch.Tensor, momentum: torch.Tensor | float, **kwargs) -> torch.Tensor:
        """
        The momentum calculation.
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True))

    def init_delta(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == "linfty":
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = clamp(delta, img_min - data, img_max - data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta: torch.Tensor, data: torch.Tensor, grad: torch.Tensor, alpha: float, **kwargs) -> torch.Tensor:
        if self.norm == "linfty":
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min - data, img_max - data)
        return delta.detach().requires_grad_(True)

    def loss_function(self, loss: str) -> nn.Module:
        """
        Get the loss function.
        """
        if loss == "crossentropy":
            return nn.CrossEntropyLoss()
        raise Exception("Unsupported loss {}".format(loss))

    def transform(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return data

    def __call__(self, *input, **kwargs) -> torch.Tensor:
        self.model.eval()
        return self.forward(*input, **kwargs)
