import torch

from sample_attacks.attack import Attack


class GIFGSM(Attack):
    """
    GI-FGSM Attack
    'Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization'(https://arxiv.org/abs/2211.11236)

    Arguments:
        model (torch.nn.Module): the surrogate model instance for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        pre_epoch (int): the pre-convergence iterations.
        s (int): the global search factor.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., pre_epoch=5, s=10
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="GI-FGSM",
        pre_epoch=5,
        s=10,
        **kwargs,
    ):
        super().__init__(attack, model, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.pre_epoch = pre_epoch
        self.s = s

    def forward(self, data: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        momentum = 0.0
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.pre_epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha * self.s)

        delta = self.init_delta(data).to(self.device)
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
        # exit()
        return delta.detach()
