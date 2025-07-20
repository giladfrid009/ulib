import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class DTLoss(torch.nn.Module):
    def __init__(self, targeted: bool, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.targeted = targeted
        self.mse_loss = ActivationLoss(loss_fn=lambda v1, v2: torch.sum(torch.square(v1 - v2), dim=1) / 2)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        clean_activ: torch.Tensor,
        adv_activ: torch.Tensor,
        clean_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.targeted:
            mse_loss = self.mse_loss(clean_activ, adv_activ)
            ce_loss = self.ce_loss(clean_logits, targets)
            return mse_loss + self.gamma * ce_loss

        # TODO: WHAT SHOULD THE LOSS BE IN THE UNTARGETED CASE?
        return self.mse_loss(clean_activ, adv_activ)


class DT_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Crafting Targeted Universal Adversarial Perturbations: Considering Images as Noise": https://ieeexplore.ieee.org/document/10323453

    Args:
        activ_extractor (ActivationExtractor): Extracts outputs of the layer preceeding the last fully-connected layer.
        alpha_step_size (float): Additive step size for the alpha parameter.
            alpha is the weight of the input images in the forward pass, and updated every batch.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        alpha_step_size: float = 1e-4,
        targeted: bool = False,
        **kwargs,
    ):
        if alpha_step_size <= 0:
            raise ValueError("`alpha_step_size` must be > 0")

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=DTLoss(targeted=targeted),
            targeted=targeted,
            **kwargs,
        )

        self.extractor = activ_extractor
        self.alpha_step_size = alpha_step_size
        self.step_num = 0

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({"attack/alpha_step_size": alpha_step_size})

    def get_alpha(self, step: int, step_size: float) -> float:
        return min(1.0, step * step_size)

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        alpha = self.get_alpha(self.step_num, step_size=self.alpha_step_size)
        self.logger.log_scalar("alpha", alpha)
        self.step_num += 1

        x_batch, y_batch = data

        with self.extractor.capture():
            logits = self.pert_model(alpha * x_batch)
            adv_activ = self.extractor.get_activations()
            self.orig_model(torch.zeros_like(x_batch))
            clean_activ = self.extractor.get_activations()

        loss = self.criterion(clean_activ, adv_activ, logits, y_batch)
        return loss
