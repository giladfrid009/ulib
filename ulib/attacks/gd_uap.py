import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class GD_UAP(OptimAttack):
    """
    ## Reference:
        Presented in "Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations": https://arxiv.org/pdf/1801.08092

    Args:
        data_dependant (bool): If True, the perturbation is computed using the input data.
            If False, the perturbation is computed using a random sample from input mean and std.
        sat_thresh (float): Threshold for the saturation rate.
            If the saturation rate is above this threshold, and changed less than `sat_delta` since the last batch,
            the perturbation is divided by 2.
        sat_delta (float): Minimum change in saturation rate since the last batch.
            If the saturation rate is above `sat_thresh` and changed less than this value,
            the perturbation is divided by 2.
    """

    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        data_dependant: bool = True,
        sat_thresh: float = 0.5,
        sat_delta: float = 0.0001,
        **kwargs,
    ):
        if sat_thresh < 0.0 or sat_thresh > 1.0:
            raise ValueError("`sat_thresh` must be in [0, 1]")

        if sat_delta < 0.0 or sat_delta > 1.0:
            raise ValueError("`sat_delta` must be in [0, 1]")

        criterion = ActivationLoss(
            loss_fn=lambda v: -torch.log(torch.sum(torch.square(v) / 2, dim=1) + torch.finfo(v.dtype).eps)
        )

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            targeted=False,
            **kwargs,
        )

        self.extractor = ActivationExtractor(self.orig_model, torch.nn.Conv2d)
        self.data_dependant = data_dependant
        self.sat_thresh = sat_thresh
        self.sat_delta = sat_delta

        self.prev_sat = 0.0
        self.curr_sat = 0.0

        self.logger.register_hparams(self.extractor.get_hparams())
        self.logger.register_hparams({"attack/data_dependant": data_dependant})
        self.logger.register_hparams({"attack/sat_thresh": sat_thresh})
        self.logger.register_hparams({"attack/sat_delta": sat_delta})

    def saturation_rate(self, pert: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sum(torch.abs(pert) == eps).float() / torch.numel(pert)

    def on_batch_start(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int):
        # Compute saturation rate
        self.prev_sat = self.curr_sat
        pert = self.pert_model.get_pert(clone=False)
        self.curr_sat = self.saturation_rate(pert, self.pert_model.eps)
        self.logger.log_scalar("saturation", self.curr_sat)

        # Divide pert if saturation rate is too high
        if self.curr_sat > self.sat_thresh and torch.abs(self.curr_sat - self.prev_sat) < self.sat_delta:
            with torch.no_grad():
                pert.divide_(2.0)
                self.curr_sat = 0.0

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor:
        x_batch, _ = data
        if self.data_dependant:
            input = x_batch
        else:
            mu = torch.mean(x_batch, dim=0, keepdim=True)
            sigma = torch.sqrt(torch.var(x_batch, dim=0, keepdim=True))
            input = mu + torch.randn_like(x_batch) * sigma

        with self.extractor.capture():
            self.pert_model(input)
            activ = self.extractor.get_activations()

        loss = self.criterion(activ)
        return loss
