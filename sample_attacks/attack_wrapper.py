from torch import Tensor
import torch
from torchattacks.attack import Attack as TorchAttack, wrapper_method
from sample_attacks.attack import Attack as BaseAttack


class AttackWrapper(TorchAttack):
    """
    A wrapper which converts `sample_attacks.attack.Attack` to `torchattacks.attack.Attack`.
    """

    def __init__(self, attack: BaseAttack):
        
        self.sample_attack = attack

        super().__init__(
            name=attack.attack,
            model=attack.model,
        )

        # TorchAttacks attributes
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False

        self.return_type = "float"
        self.normalization_used = None
        self._normalization_applied = False

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        self.sample_attack.get_logits(inputs, *args, **kwargs)

    @wrapper_method
    def set_device(self, device):
        self.sample_attack.device = device
        return super().set_device(device)

    @wrapper_method
    def set_model(self, model):
        self.sample_attack.model = model
        super().set_model(model)

    def set_normalization_used(self, mean, std):
        """Not Supported."""
        raise NotImplementedError()

    def normalize(self, inputs):
        """Not Supported."""
        raise NotImplementedError()

    def inverse_normalize(self, inputs):
        """Not Supported."""
        raise NotImplementedError()

    def set_model_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        """Not Supported."""
        raise NotImplementedError()

    def forward(self, inputs: Tensor, labels: Tensor, *args, **kwargs) -> Tensor:
        """
        The general attack procedure

        Arguments:
            inputs (N, C, H, W): tensor for input images in [0, 1]
            labels (N,): tensor for ground-truth labels if untargeted
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
            
        Returns: (N, C, H, W) tensor for adversarial images
        """
        pert = self.sample_attack.forward(inputs, labels, *args, **kwargs)
        return torch.clamp(inputs + pert, min=0, max=1).detach()
