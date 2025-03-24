import torch
from ulib.pert_module import PertModule
from ulib.attack import UnivAttack
import random


class DU_ATTACK(UnivAttack):
    """
    Reference:
        Presented in "Decision-based Universal Adversarial Attack": https://arxiv.org/pdf/2009.07024
        Code adapted from original peper repo: https://github.com/JingWu321/DUAttack/tree/master
        Note: Official code differs from the algorithm presented in the paper, here we use the code implementation.
    """

    def __init__(
        self,
        pert_model: PertModule,
        alpha: float = 0.2,
        beta: float = 0.9,
        steps: int = 10,
        **kwargs,
    ):
        if alpha <= 0:
            raise ValueError("Learning rate must be > 0")
        
        if beta < 0:
            raise ValueError("Momentum factor must be >= 0")
        
        super().__init__(
            pert_model=pert_model,
            **kwargs,
        )

        self.alpha = alpha
        self.beta = beta
        self.steps = steps

        self.logger.register_hparams({"attack/alpha": alpha})
        self.logger.register_hparams({"attack/beta": beta})
        self.logger.register_hparams({"attack/steps": steps})

    def generate_noise(self) -> torch.Tensor:
        C, H, W = self.pert_model.data_shape
        noise = torch.zeros(self.pert_model.shape, dtype=self.pert_model.dtype, device=self.pert_model.device)
        rnd_channel = random.randint(0, C - 1)
        shift = random.randint(0, max(H, W) - 1)
        channel_noise = self.shifted_eye(shift, size=(H, W), dtype=noise.dtype)
        noise[:, rnd_channel, :, :] = channel_noise.to(noise.device)
        return noise

    def shifted_eye(self, shift: int, size: tuple[int, int], dtype=None) -> torch.Tensor:
        H, W = size
        skewed_eye = torch.zeros((H, W), dtype=dtype)
        row_indices = torch.arange(H)
        col_indices = (row_indices + shift) % W
        skewed_eye[row_indices, col_indices] = 1
        return skewed_eye

    @torch.no_grad()
    def process_batch(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> float | None:
        x_batch, y_batch = data

        corr_left = 1.0
        corr_right = 1.0
        pert = self.pert_model.get_pert()
        momentum = torch.zeros_like(self.pert_model.get_pert())

        for step in range(self.steps):
            noise = self.generate_noise()

            pert_left = pert - (self.alpha + self.beta * momentum) * noise
            pert_left = self.pert_model.project(pert_left)

            input = self.pert_model.clamp_inputs(pert_left + x_batch)
            preds = self.orig_model(input).argmax(dim=1)
            corr_left = (preds == y_batch).float().mean().item()

            if (self.targeted and corr_left == 1.0) or (not self.targeted and corr_left == 0.0):
                pert = pert_left
                break

            pert_right = pert + (self.alpha + self.beta * momentum) * noise
            pert_right = self.pert_model.project(pert_right)

            input = self.pert_model.clamp_inputs(pert_right + x_batch)
            preds = self.orig_model(input).argmax(dim=1)
            corr_right = (preds == y_batch).float().mean().item()

            if (self.targeted and corr_right > corr_left) or (not self.targeted and corr_right < corr_left):
                pert = pert_right
                momentum = momentum + self.alpha * noise
            else:
                pert = pert_left
                momentum = momentum - self.alpha * noise

        self.pert_model.set_pert(pert)

        return min(corr_left, corr_right)
