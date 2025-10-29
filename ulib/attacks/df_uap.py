import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class LogitsMargin(torch.nn.Module):
    def __init__(self, kappa: float = 0.0, targeted: bool = False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        one_hots = torch.nn.functional.one_hot(y, num_classes=logits.size(1))
        logits_target = torch.sum(one_hots * logits, dim=1)
        logits_masked = torch.masked_fill(logits, one_hots == 1, float("-inf"))
        logits_others, _ = torch.max(logits_masked, dim=1)

        if self.targeted:
            logits_others = torch.detach(logits_others)
            loss = torch.clamp(logits_others - logits_target, min=-self.kappa)
        else:
            loss = torch.clamp(logits_target - logits_others, min=-self.kappa)

        return loss.mean()


class DF_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations": https://arxiv.org/pdf/2007.06189

    Args:
        kappa (float): Minimum margin between the target and the maximum non-targeted logit.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        kappa: float = 0.0,
        targeted: bool = False,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=LogitsMargin(kappa=kappa, targeted=targeted),
            targeted=targeted,
            **kwargs,
        )

    def compute_loss(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> torch.Tensor:
        x_batch, y_batch = data
        preds = self.pert_model(x_batch)
        loss = self.criterion(preds, y_batch)
        return loss
