import torch
import torchattacks.attack
from typing import Iterable
from ulib.attack import UnivAttack
from ulib.pert_module import PertModule
from ulib.activation_extractor import ActivationExtractor, ActivationLoss


class IML_SVRG_UAP(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        learning_rate: float,
        inner_attack: torchattacks.attack.Attack,
        criterion: ActivationLoss,
        activ_extractor: ActivationExtractor,
        num_batches: int,
        y_s_eps: float = 8 / 255,
        mode: str = "classic", # "classic" | "averge" | "α-svrg"
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            targeted=False,
            **kwargs,
        )
        self.mode = mode

        self.inner_attack = inner_attack
        self.criterion = criterion
        self.extractor = activ_extractor

        self.num_batches = num_batches
        self.y_s = PertModule(pert_model.model, data_shape=pert_model.data_shape, eps=y_s_eps).to(self.device)
        self.lr = learning_rate

        self.pert_model.to(self.device)

        self.grad_F_y = torch.zeros_like(self.pert_model.get_pert(), device=self.device)
        self.batch_grads: torch.Tensor = torch.zeros((num_batches, *self.pert_model.get_pert().shape), device=self.device)

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({"attack/criterion": criterion.__class__.__name__})
        self.logger.register_hparams({f"attack/inner_attack/{k}": v for k, v in inner_attack.__dict__.items()})
        self.logger.register_hparams({"attack/inner_attack/name": inner_attack.__class__.__name__})
        self.logger.register_hparams({"attack/learning_rate": learning_rate})
        self.logger.register_hparams({"attack/y_s_eps": y_s_eps})

    def on_epoch_start(self, dl_train: Iterable[tuple[torch.Tensor, ...]], epoch_num: int):
        super().on_epoch_start(dl_train, epoch_num)
        self.batch_grads.zero_()
        self.grad_F_y = self._calc_full_grad(dl_train, self.y_s)

    def on_epoch_end(self, epoch_num: int):
        super().on_epoch_end(epoch_num)
        match self.mode:
            case "classic":
                self.y_s.set_pert(self.batch_grads[-1,...])
            case "average":
                self.grad_F_y = self.batch_grads.mean(dim=0)
            case "α-svrg":
                self.grad_F_y = self.batch_grads.mean(dim=0)
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")
        # self.y_s.set_pert(self.batch_grads.mean(dim=0))

    def _calc_grad(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        model: PertModule,
    ) -> torch.Tensor:
        activ, adv_act = self._process_batch(x_batch, y_batch, model)
        if activ is None or adv_act is None:
            return torch.zeros_like(model.get_pert(), device=self.device)
        loss = self.criterion(activ, adv_act)
        grad = torch.autograd.grad(loss, model._pert, allow_unused=True)[0]
        return grad

    def _calc_full_grad(
        self,
        dl_train: Iterable[tuple[torch.Tensor, ...]],
        model: PertModule,
    ) -> torch.Tensor:
        full_grad = torch.zeros_like(model.get_pert(), device=self.device)
        num_samples = 0
        for x_batch, y_batch in dl_train:
            num_samples += y_batch.size(0)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            activ, adv_act = self._process_batch(x_batch, y_batch, model)
            if activ is None or adv_act is None:
                continue
            loss = self.criterion(activ, adv_act)
            full_grad += torch.autograd.grad(loss, model._pert)[0]
            del x_batch, y_batch, activ, adv_act

        return full_grad  / num_samples# / self.num_batches

    def _process_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor, model: PertModule) -> tuple[torch.Tensor, torch.Tensor]:
        with self.extractor.capture():
            # record forward pass activations
            y_pred = model(x_batch).argmax(dim=1)
            activ = self.extractor.get_activations()

            # attack only correctly classified samples
            cor_mask = y_pred == y_batch
            if not cor_mask.any():
                return None
            x_batch = x_batch[cor_mask]
            y_batch = y_batch[cor_mask]
            activ = {k: v[cor_mask] for k, v in activ.items()}

            # per-sample attack
            pert = model.get_pert()
            x_pert = model.clamp_inputs(x_batch + pert)
            x_attk = self.inner_attack.forward(x_pert.detach(), y_batch)

            # record forward pass activations
            y_attk = self.orig_model(x_attk).argmax(dim=1)
            adv_act = self.extractor.get_activations()

            # use only successfully attacked samples
            suc_mask = y_attk != y_batch
            if not suc_mask.any():
                return None
            activ = {k: v[suc_mask] for k, v in activ.items()}
            adv_act = {k: v[suc_mask] for k, v in adv_act.items()}

        return activ, adv_act

    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
    ) -> float:
        x_batch, y_batch = data
        self.curr_epoch = epoch_num
        self.curr_bach = batch_num

        activ, adv_act = self._process_batch(x_batch, y_batch, self.pert_model)
        if activ is None or adv_act is None:
            return None
        # Maximize similarity between perturbed and adversarial inputs
        loss = self.criterion(activ, adv_act)
        grad = torch.autograd.grad(loss, self.pert_model._pert)[0]
        # grad = self._calc_grad(x_batch, y_batch, self.pert_model)
        grad_f_y = self._calc_grad(x_batch, y_batch, self.y_s)
        self.pert_model._pert.data += self.lr * (grad - grad_f_y + self.grad_F_y)

        self.batch_grads[batch_num, ...] = self.pert_model.get_pert().detach()
        return loss.item()
