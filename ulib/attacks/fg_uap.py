import torch
from ulib.pert_module import PertModule
from ulib.attack import OptimAttack
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class FG_UAP(OptimAttack):
    """
    Reference:
        Presented in "FG-UAP: Feature-Gathering Universal Adversarial Perturbation": https://arxiv.org/pdf/2209.13113
        
    Args:
        activ_extractor (ActivationExtractor): Extracts outputs of the preceding layer to the last fully-connected layer.
        skip_already_fooled (bool): Skip samples that are already fooled by the current perturbation.
    """
    def __init__(
        self,
        pert_model: PertModule,
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        skip_already_fooled: bool = False,
        **kwargs,
    ):
        criterion = ActivationLoss(loss_fn=lambda v1, v2: torch.cosine_similarity(v1, v2, dim=-1))

        super().__init__(
            pert_model=pert_model,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs,
        )

        self.extractor = activ_extractor
        self.skip_already_fooled = skip_already_fooled
        
        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({"attack/skip_already_fooled": skip_already_fooled})
        
    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor | None:
        x_batch, y_batch = data
        with self.extractor.capture():
            
            preds = self.pert_model(x_batch)
            adv_activ = self.extractor.get_activations()
            
            if self.skip_already_fooled:
                cor_mask = preds.argmax(dim=-1) == y_batch
                if not cor_mask.any():
                    return None
                x_batch = x_batch[cor_mask]
                adv_activ = {k: v[cor_mask] for k, v in adv_activ.items()}
            
            self.orig_model(x_batch)
            clean_activ = self.extractor.get_activations()
            
        loss = self.criterion(clean_activ, adv_activ)
        if self.targeted:
            loss = -loss
        return loss