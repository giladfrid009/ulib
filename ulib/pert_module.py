from __future__ import annotations
import torch
from torch import nn
from typing import Iterator
from ulib import utils


class PertModule(nn.Module):
    """
    A PyTorch module that applies a trainable adversarial perturbation to input data and passes it through a pretrained model.
    The model is kept in evaluation mode.

    ## Reference:
        Inspired by MIT-LL RAI [https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/]
    """

    def __init__(
        self,
        model: nn.Module,
        data_shape: tuple[int, ...],
        eps: float,
        norm: float = float("inf"),
        random_init: bool = True,
        input_range: tuple[float, float] | tuple[torch.Tensor, torch.Tensor] = (0, 1),
        input_clamp: bool = True,
    ) -> None:
        """
        Initialize the PertModule.

        Args:
            model (nn.Module): The attacked pertrained model.
            data_shape (tuple[int, ...]): Shape of the input data.
            eps (float): Maximum allowable magnitude of the perturbation.
            norm (float): Perturbation norm constraint; must be >= 1, supports float("inf") as infinity norm.
            random_init (bool): If True, randomly initializes the perturbation within the epsilon-ball.
            input_range (tuple[float, float] | tuple[torch.Tensor, torch.Tensor]): Input data range.
            input_clamp (bool): If True, clamps the input data to the input_range after applying the perturbation.
        """
        super().__init__()

        if norm < 1:
            raise ValueError("Norm must be at least 1.")
        if eps <= 0:
            raise ValueError("Epsilon must positive")

        self.model = model.eval()
        self.model.requires_grad_(False)

        self.device = utils.extract_device(model)
        self.data_shape = data_shape
        self.eps = eps
        self.norm = norm
        self.rnd_init = random_init
        self.input_range = input_range
        self.input_clamp = input_clamp

        self._pert = nn.Parameter(torch.zeros(1, *data_shape, device=self.device))

        if self.rnd_init:
            self.random_init()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._pert.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._pert.dtype

    def get_hparams(self) -> dict[str, object]:
        """
        Get the hyperparameters of the perturbation module.

        Returns:
            dict[str, object]: Dictionary of hyperparameters.
        """
        return {
            "pert/name": self.__class__.__name__,
            "pert/eps": self.eps,
            "pert/norm": self.norm,
            "pert/shape": str(self.shape),
            "pert/dtype": str(self.dtype),
            "pert/random_init": self.rnd_init,
            "pert/input_range": str(self.input_range),
            "pert/input_clamp": self.input_clamp,
        }

    @torch.no_grad()
    def set_pert(self, pert: torch.Tensor) -> None:
        """
        Set the perturbation to a copy of the given tensor.

        Args:
            pert (torch.Tensor): Tensor containing the new perturbation values.
        """
        if pert.shape not in {self._pert.shape, self.data_shape}:
            raise ValueError("Invalid perturbation shape.")

        pert = pert.view_as(self._pert).to(self.device)
        self._pert.copy_(pert)

    @torch.no_grad()
    def get_pert(self, clone: bool = True) -> torch.Tensor:
        """
        Get the perturbation parameter tensor, projected onto the epsilon-ball.
        It is highly recommended to access the perturbation through this method.

        Args:
            clone (bool): If True, returns a cloned and detached tensor, otherwise returns the original tensor.

        Returns:
            torch.Tensor: The projected perturbation tensor.
        """
        pert = self.project(self._pert)

        if clone:
            return pert.clone().detach()
        else:
            self._pert.copy_(pert)
            return self._pert

    @torch.no_grad()
    def random_init(self) -> None:
        """
        Randomly initialize the perturbation within the valid epsilon-ball.
        """
        if self.norm == float("inf"):
            rnd = torch.empty_like(self._pert).uniform_(-self.eps, self.eps)
        elif self.norm >= 1:
            rnd = utils.sample_lp_ball(self._pert.numel(), self.norm, device=self.device)
            rnd = (rnd * self.eps).view_as(self._pert)
        else:
            raise ValueError(f"Invalid norm value: {self.norm}")

        self._pert.copy_(self.project(rnd))

    def clamp_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp the input tensor to the expected input range if `input_clamp` is True.
        Otherwise, returns the input tensor as is.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Clamped input tensor.
        """
        if not self.input_clamp:
            return x
        lb, ub = self.input_range
        return torch.clamp(x, lb, ub)

    def project(self, pert: torch.Tensor) -> torch.Tensor:
        """
        Project the perturbation tensor onto the epsilon-ball.

        Args:
            pert (torch.Tensor): Perturbation tensor to project.

        Returns:
            torch.Tensor: Projected perturbation tensor.
        """
        if self.norm == float("inf"):
            return torch.clamp(pert, -self.eps, self.eps)
        elif self.norm >= 1:
            flat_pert = pert.view(1, -1)
            projected = torch.renorm(flat_pert, p=self.norm, dim=1, maxnorm=self.eps)
            return projected.view_as(pert)
        else:
            raise ValueError(f"Invalid norm value: {self.norm}")

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Yields the trainable parameters of the module. Only the perturbation parameter is trainable.

        Yields:
            Iterator[nn.Parameter]: Trainable perturbation parameter.
        """
        yield self._pert

    def train(self, mode: bool = True) -> PertModule:
        """
        Overrides the default train() method to ensure the inner model remains in evaluation mode.

        Args:
            mode (bool): Training mode flag (True for training, False for evaluation).

        Returns:
            PertModule: Self.
        """
        self.training = mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the perturbation to the input and pass it through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        with torch.no_grad():
            self._pert.copy_(self.project(self._pert))
        x = x.to(self.device)
        x_adv = x + self._pert
        x_adv = self.clamp_inputs(x_adv)
        return self.model(x_adv)

    @torch.no_grad()
    def to_image(self) -> torch.Tensor:
        """
        Convert the current perturbation into an image format for visualization.

        Returns:
            torch.Tensor: Normalized image tensor with values in the range [0, 1].
        """
        pert = self.project(self._pert).detach().cpu()
        pert = pert.view(*self.data_shape)
        # Scale perturbation to be in the Lp unit ball, normalize to [-1, 1] and map to [0, 1]
        pert = pert / self.eps
        pert = torch.clamp(pert, -1, 1)
        pert = pert / 2 + 0.5
        return torch.clamp(pert, 0, 1)
