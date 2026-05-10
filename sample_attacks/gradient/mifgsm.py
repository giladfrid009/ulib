import torch

from sample_attacks.attack import Attack


class MIFGSM(Attack):
    """
    MI-FGSM Attack
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)

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
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
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
        attack="MI-FGSM",
        **kwargs,
    ):
        super().__init__(attack, model, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
