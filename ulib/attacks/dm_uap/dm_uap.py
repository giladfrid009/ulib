import torch
import copy
from torch import nn
from torch import optim

from ulib.pert_module import PertModule
from ulib.attack import UnivAttack
from ulib.attacks.dm_uap.pgd_l2 import PGD_Theta_Apart


class DM_UAP(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        total_epochs: int = 20,
        beta: float = 9.0,
        theta_neighborhood: float = 1.0,
        data_neighborhood: float = 32,
        data_steps: int = 10,
        theta_steps: int = 10,
        smooth_rate: float = 0.2,
        **kwargs,
    ):
        """
        ## Reference:
            Presented in "Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization": https://arxiv.org/pdf/2503.12793

        Code Reference: https://github.com/yechao-zhang/DM-UAP

        Args:
            total_epochs (int): Number of training epochs (T in paper).
            beta (float): Loss clamping value.
            theta_neighborhood (float): Model maximum neighborhood size (Rho in paper).
            data_neighborhood (float): Data maximum neighborhood size (r in paper).
            data_steps (int): Number of data optimization steps in the inner PGD attack (Kd in paper).
            theta_steps (int): Number of steps of model param optimization in the inner PGD attack (Km in paper).
            smooth_rate (float):
        """
        super().__init__(pert_model=pert_model, **kwargs)

        self.total_epochs = total_epochs
        self.beta = beta
        self.theta_neighborhood = theta_neighborhood
        self.data_neighborhood = data_neighborhood
        self.data_steps = data_steps
        self.theta_steps = theta_steps
        self.smooth_rate = smooth_rate

        self.orig_state_dict = copy.deepcopy(self.orig_model.state_dict())
        self.optimizer = optim.AdamW(self.pert_model.parameters(), lr=0.1)

        self.inner_attack = self.make_attack(0)

        self.metric_tracker.report_hparams(
            "attack",
            total_epochs=self.total_epochs,
            beta=self.beta,
            rho=self.theta_neighborhood,
            aa=self.data_neighborhood,
            data_steps=self.data_steps,
            theta_steps=self.theta_steps,
            smooth_rate=self.smooth_rate,
        )

    def clamped_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = torch.mean(torch.clamp(loss_fn(output, target), max=self.beta))  # type: ignore
        return loss

    def make_attack(self, epoch_num: int) -> PGD_Theta_Apart:
        rho_step = self.theta_neighborhood / self.total_epochs
        eps_step = self.data_neighborhood / self.total_epochs
        rho_current = rho_step * (epoch_num + 1)
        eps_current = eps_step * (epoch_num + 1)

        self.inner_attack = PGD_Theta_Apart(
            eps=eps_current,
            rho=rho_current,
            data_steps=self.data_steps,
            theta_steps=self.theta_steps,
            random_start=False,
            eps_for_division=1e-10,
            smooth_rate=self.smooth_rate,
        )

        return self.inner_attack

    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
        step_num: int,
    ) -> float | None:

        if batch_num == 0 and epoch_num > 0:  # on epoch start
            self.make_attack(epoch_num)

        x_batch, _ = data
        model = self.orig_model

        with torch.no_grad():
            outputs_ori = model(x_batch)
            _, target_label = torch.max(outputs_ori, dim=1)

        self.optimizer.zero_grad()
        model, x_val = self.inner_attack.forward(model, x_batch, target_label)

        pert = self.pert_model.get_pert(clone=False)
        outputs = model(x_val + pert)

        loss = -self.clamped_loss(outputs, target_label)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pert = self.pert_model.project(pert)
            self.pert_model.set_pert(pert)

        model.load_state_dict(self.orig_state_dict)
        model.eval()

        return loss.item()
