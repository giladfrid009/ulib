import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class DTLoss(torch.nn.Module):
    def __init__(self, targeted: bool, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.targeted = targeted
        self.mse_loss = ActivationLoss(loss_fn=lambda v1, v2: torch.sum(torch.square(v1.flatten(1) - v2.flatten(1)), dim=1) / 2)
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
        alpha_step_size (float): Additive step size for the alpha parameter.
            alpha is the weight of the input images in the forward pass, and updated every batch.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        alpha_step_size: float = 1e-4,
        targeted: bool = True,
        **kwargs,
    ):
        if alpha_step_size <= 0:
            raise ValueError("`alpha_step_size` must be > 0")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=DTLoss(targeted=targeted, gamma=1.0),
            targeted=targeted,
            **kwargs,
        )

        self.extractor = activ_extractor
        self.alpha_step_size = alpha_step_size
        self.step_num = 0

        self.metric_tracker.report_hparams("activ_extractor", activ_extractor.get_hparams())
        self.metric_tracker.report_hparams("attack", alpha_step_size=alpha_step_size)

    def get_alpha(self, step: int, step_size: float) -> float:
        return min(1.0, step * step_size)

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor:
        alpha = self.get_alpha(self.step_num, step_size=self.alpha_step_size)
        self.metric_tracker.report_scalar("alpha", alpha, step_num)
        self.step_num += 1

        x_batch, y_batch = data

        with self.extractor.capture():
            adv_logits = self.pert_model(alpha * x_batch)
            adv_activ = self.extractor.get_activations()
            self.pert_model(torch.zeros_like(x_batch))
            uap_activ = self.extractor.get_activations()

        loss = self.criterion(uap_activ, adv_activ, adv_logits, y_batch)
        return loss
