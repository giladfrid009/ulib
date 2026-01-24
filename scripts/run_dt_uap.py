from argparse import ArgumentParser
import sys
import pathlib
import torch

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from ulib import ActivationExtractor
from ulib.attacks.dt_uap import DT_UAP


class DT_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        args = parser.add_argument_group("DT-UAP Attack Arguments")
        
        args.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="Learning rate for Adam optimizer",
        )
        
        args.add_argument(
            "--layer_name",
            type=str,
            default="relu",
            help="The name of the layer to extract activations from.",
        )
        
        args.add_argument(
            "--alpha_step_size",
            type=float,
            default=1e-3,
            help="Additive step size for the alpha parameter.",
        )
        
        args.add_argument(
            "--targeted",
            action="store_true",
            help="If set, the attack is targeted.",
        )

    def initialize_attack(self, pert_model, evaluator, eval_freq, mixed_precision, metric_tracker):
        args = self.args()
        optimizer = torch.optim.Adam(pert_model.parameters(), lr=args.lr, weight_decay=1e-10) # weight_decay from notebook
        
        extractor = ActivationExtractor(pert_model.model, args.layer_name, exact_match=True)

        return DT_UAP(
            pert_model=pert_model,
            optimizer=optimizer,
            activ_extractor=extractor,
            alpha_step_size=args.alpha_step_size,
            targeted=args.targeted,
            # general args
            eval_freq=eval_freq,
            evaluator=evaluator,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    DT_Experiment().main()
