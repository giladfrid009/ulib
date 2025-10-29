import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack


class GradSign(torch.optim.Optimizer):
    """
    A PyTorch optimizer that applies sign to the gradients of the model's parameters before applying the internal optimizer step.
    This optimizer is intended to be used as a wrapper around an existing optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.state = optimizer.state

    def step(self, closure=None):  # type: ignore
        # Normalizes all gradients to be sign vectors
        # Before applying the internal optimizer step

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad.sign_()

        return self.optimizer.step(closure)


class UFGSM(OptimAttack):
    """
    Universal Fast Gradient Sign Method (UFGSM) attack.
    Applies `torch.sign` to the gradients of the model's parameters before applying the internal optimizer step.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            optimizer=GradSign(optimizer),
            criterion=criterion,
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
        if self.targeted:
            loss = -loss
        return loss
