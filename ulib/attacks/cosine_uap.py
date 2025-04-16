import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class CosineSimilarity(torch.nn.Module):
    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        # we want to minimize this loss
        v1 = v1.view(v1.size(0), -1)
        v2 = v2.view(v2.size(0), -1)
        return torch.cosine_similarity(v1, v2, dim=1).mean()


class Cosine_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Data-free Universal Adversarial Perturbation and Black-box Attack": https://ieeexplore.ieee.org/document/9710529
    """
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=CosineSimilarity(),
            **kwargs,
        )

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        x_batch, _ = data
        preds_adv = self.pert_model(x_batch)
        preds_clean = self.orig_model(x_batch)
        loss = self.criterion(preds_adv, preds_clean)
        if self.targeted:
            loss = -loss
        return loss
