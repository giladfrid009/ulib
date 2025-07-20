import torch
from tqdm import tqdm
from ulib.attacks.spm.power_method import PowerMethod, JacobianOperator


class SPM_UAP:
    """
    ## Reference:
        Presented in "Art of singular vectors and universal adversarial perturbations": https://arxiv.org/pdf/1709.03582
        Code taken from: https://github.com/slayff/art_of_vectors_pytorch
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        p: float = float("inf"),
        q: float = 10,
        pm_maxiter: int = 20,
        device=torch.device("cpu"),
        verbose=0,
    ):
        self.input_shape = input_shape
        self.input_dim = torch.prod(torch.tensor(input_shape)).item()
        self.hidden_dim = torch.prod(torch.tensor(output_shape)).item()
        self.power_method = PowerMethod(p, q, maxiter=pm_maxiter, device=device, verbose=verbose)
        self.device = device
        self.verbose = verbose

    def fit(self, mfe, img_iter, n_batches=1):
        mfe.to(self.device)
        mfe.eval()

        for i, img_data in enumerate(img_iter):
            if i >= n_batches:
                break

            if self.verbose:
                print(f"Running power method on batch #{i}")

            img_batch, _ = img_data
            img_batch = img_batch.to(self.device)
            jac = JacobianOperator(img_batch, mfe, self.input_dim, self.hidden_dim, self.device)
            self.power_method.fit(jac)

    def predict_raw(self, mfe, img_iter):
        mfe.to(self.device)
        mfe.eval()

        probs = []
        preds = []
        for i, img_data in enumerate(tqdm(img_iter, disable=not self.verbose)):
            img_batch, _ = img_data
            img_batch.to(self.device)
            probabilities = torch.softmax(mfe(img_batch), dim=-1)
            cur_probs, cur_preds = torch.max(probabilities, dim=-1)

            probs.append(cur_probs)
            preds.append(cur_preds)

        return dict(predictions=torch.cat(preds, dim=0), probabilities=torch.cat(probs, dim=0))

    def get_perturbation(self, eps: float):
        return self.power_method.get_perturbation(self.input_shape, eps=eps)

    @staticmethod
    def fooling_rate(model_raw_pred, model_pert_pred):
        return (model_raw_pred != model_pert_pred).float().mean().item()
