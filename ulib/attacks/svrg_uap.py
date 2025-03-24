import torch
from torch import nn
import copy
from typing import Iterable
from ulib.pert_module import PertModule
from ulib.attack import UnivAttack
from ulib.pert_module import PertModule


class SVRG_UAP(UnivAttack):
    def __init__(
        self,
        pert_model: PertModule,
        criterion: nn.Module,
        num_batches: int,
        learning_rate: float,
        y_s_eps: float = 8 / 255,
        attk_correct: bool = True,
        mode: str = "classic", # "classic" | "averge" | "α-svrg" | "supernova"
        **kwargs,
    ):
        super().__init__(
            pert_model=pert_model,
            targeted=False,
            **kwargs,
        )
        self.mode = mode
        self.curr_epoch = 0
        self.curr_bach = 0

        self.optim = torch.optim.SGD(pert_model.parameters(), lr=learning_rate)
        self.criterion = criterion

        self.num_batches = num_batches
        self.y_s = PertModule(pert_model.model, data_shape=pert_model.data_shape, eps=y_s_eps).to(self.device)
        self.lr = learning_rate
        self.attk_correct = attk_correct

        self.pert_model.to(self.device)

        self.grad_F_y = torch.zeros_like(self.pert_model.get_pert(), device=self.device)
        self.batch_grads: torch.Tensor = torch.zeros((num_batches, *self.pert_model.get_pert().shape), device=self.device)

        # self.add_hparams(utils.extract_optim_hparams(optim))
        self.logger.register_hparams({"attack/criterion": criterion.__class__.__name__})
        self.logger.register_hparams({"attack/learning_rate": learning_rate})
        self.logger.register_hparams({"attack/y_s_eps": y_s_eps})

    def _process_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor, model: PertModule) -> tuple[torch.Tensor, torch.Tensor]:
        preds = model(x_batch)

        # only attack correctly classified samples
        if self.attk_correct:# and self.curr_epoch % 3 != 0:
            cor_mask = preds.argmax(dim=1) == y_batch
            if not cor_mask.any():
                return None, None
            x_batch = x_batch[cor_mask]
            y_batch = y_batch[cor_mask]
            preds = preds[cor_mask]

        return preds, y_batch

    def _calc_grad(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        model: PertModule,
    ) -> torch.Tensor:
        activ, adv_act = self._process_batch(x_batch, y_batch, model)
        if activ is None or adv_act is None:
            return torch.zeros_like(model.get_pert(), device=self.device)
        loss = self.criterion(activ, adv_act) / x_batch.size(0)
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
            x_batch, y_batch = self.to_device(x_batch, y_batch)
            activ, adv_act = self._process_batch(x_batch, y_batch, model)
            if activ is None or adv_act is None:
                continue
            loss = self.criterion(activ, adv_act)
            full_grad += torch.autograd.grad(loss, model._pert)[0]
            del x_batch, y_batch, activ, adv_act

        return full_grad  / num_samples# / self.num_batches

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
            case "supernova":
                self.grad_F_y = self.batch_grads.mean(dim=0)
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")
        # self.y_s.set_pert(self.batch_grads.mean(dim=0))
        print()

    def process_batch(
        self,
        data: tuple[torch.Tensor, ...],
        batch_num: int,
        epoch_num: int,
    ) -> float | None:
        self.curr_epoch = epoch_num
        self.curr_bach = batch_num

        x_batch, y_batch = data

        activ, adv_act = self._process_batch(x_batch, y_batch, self.pert_model)
        if activ is None or adv_act is None:
            return None
        # Maximize similarity between perturbed and adversarial inputs
        loss = self.criterion(activ, adv_act) / x_batch.size(0)
        grad = torch.autograd.grad(loss, self.pert_model._pert)[0]
        # grad = self._calc_grad(x_batch, y_batch, self.pert_model)

        grad_f_y = self._calc_grad(x_batch, y_batch, self.y_s)
        svrg_grad = grad - grad_f_y + self.grad_F_y
        # svrg_grad = self.pert_model.eps * torch.sign(svrg_grad)

        lr = self.optim.param_groups[0]["lr"]
        self.logger.log_scalar("lr", lr)

        self.pert_model._pert.data += lr * svrg_grad
        if self.mode == 'supernova':
            self.batch_grads[batch_num, ...] = grad.clone().detach()
        else:
            self.batch_grads[batch_num, ...] = self.pert_model.get_pert().detach()
        # self.batch_grads[batch_num, ...] = grad.detach().clone()
        return loss.item()
