import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class DTLoss(torch.nn.Module):
    def __init__(self, targeted: bool, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.targeted = targeted
        self.mse_loss = ActivationLoss(loss_fn=lambda v1, v2: torch.mean(torch.square(v1.flatten(1) - v2.flatten(1)), dim=1), reduction="mean")
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        uap_activ: torch.Tensor,
        adv_activ: torch.Tensor,
        adv_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for the DT_UAP attack.

        Args:
            uap_activ (torch.Tensor): Activations only from the UAP, `f_l(v)` from the paper.
            adv_activ (torch.Tensor): Activations from the perturbed inputs, `f_l(alpha * x + v)` from the paper.
            adv_logits (torch.Tensor): Logits from the perturbed inputs, `f(alpha * x + v)` from the paper.
            targets (torch.Tensor): Target labels for the attack, `y_target` from the paper.
        """
        if self.targeted:
            mse_loss = self.mse_loss(uap_activ, adv_activ)
            ce_loss = self.ce_loss(adv_logits, targets)
            return mse_loss + self.gamma * ce_loss

        return self.mse_loss(uap_activ, adv_activ)


class DT_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Crafting Targeted Universal Adversarial Perturbations: Considering Images as Noise": https://ieeexplore.ieee.org/document/10323453

    Args:
        activ_extractor (ActivationExtractor): Extracts outputs of the layer preceding the last fully-connected layer.
        alpha (float): The weight of the input images in the forward pass.
        alpha_step (float | None): If not `None`, the value of `alpha` will be linearly increased from 0 to `alpha`, with a step size of `alpha_step`.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        alpha: float = 0.3,
        alpha_step: float | None = 0.0001,
        targeted: bool = True,
        **kwargs,
    ):
        if not (0 <= alpha <= 1):
            raise ValueError("`alpha` must be in [0, 1]")

        if alpha_step is not None and not (0 <= alpha_step <= 1):
            raise ValueError("`alpha_step` must be in [0, 1]")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=DTLoss(targeted=targeted, gamma=1.0),
            targeted=targeted,
            **kwargs,
        )

        self.extractor = activ_extractor
        self.alpha = alpha
        self.alpha_step = alpha_step

        self.metric_tracker.report_hparams("activ_extractor", activ_extractor.get_hparams())
        self.metric_tracker.report_hparams("attack", alpha=alpha, alpha_step=alpha_step)

    def compute_alpha(self, step: int) -> float:
        if self.alpha_step is None:
            return self.alpha

        return min(self.alpha, self.alpha_step * (step + 1))

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor:

        x_batch, y_batch = data

        with self.extractor.capture():
            alpha = self.compute_alpha(step_num)
            self.metric_tracker.report_scalar("DT-UAP/alpha", alpha, step_num)

            adv_logits = self.pert_model(alpha * x_batch)
            adv_activ = self.extractor.get_activations()
            self.pert_model(torch.zeros_like(x_batch))
            uap_activ = self.extractor.get_activations()

        loss = self.criterion(uap_activ, adv_activ, adv_logits, y_batch)
        return loss
