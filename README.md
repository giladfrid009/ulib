# Universal Adversarial Perturbations Library (ULib)

ULib is a PyTorch-based library designed for generating universal adversarial perturbations (UAPs) to fool deep neural networks. It provides a modular and extensible framework for implementing and evaluating various UAP generation techniques. This README provides a comprehensive guide to using the library, its structure, and key components.

## Features

-   **Modular Attack Implementations:** Includes implementations of various UAP generation methods, such as GD\_UAP, DF\_UAP, DT\_UAP, IMI\_UAP, IML\_SVRG\_UAP, FFF, UAPGD, FG\_UAP, SD\_UAP, Cosine UAP, AE\_UAP, D-BADGE and more.
-   **Flexible Perturbation Module:** The [`PertModule`](ulib/pert_module.py) wraps the target model, allowing for easy integration and control of adversarial perturbations.
-   **Extensive Experiment Notebooks:** A collection of Jupyter notebooks demonstrating the usage of different attacks on various datasets and models.
-   **Built-in Logging:** Integrated logging using the [`Logger`](ulib/logger.py) class for tracking metrics, hyperparameters, and generated adversarial examples.
-   **Stop Criteria:** Flexible stopping criteria for attacks, including time limits and early stopping.
-   **Data Extraction Tools:** Utilities for splitting and filtering datasets based on correctness, class label, or confidence.

## Repository Structure

```
.
├── .gitignore       
├── LICENSE           
├── README.md         
├── requirements.yaml 
├── notebooks/                # Jupyter notebooks demonstrating library usage
│   ├── datasets.py           # Utility functions for loading datasets
│   ├── experiment_robust.py  # Functions for setting up robust training experiments
│   ├── experiment_torch.py   # Functions for setting up standard PyTorch experiments
│   ├── experiment_utils.py   # Utility functions for experiments
│   ├── test_df_uap.ipynb     # Notebook testing the DF_UAP attack
│   ├── test_dt_uap.ipynb     # Notebook testing the DT_UAP attack
│   ├── test_fff.ipynb        # Notebook testing the FFF attack
│   └── ...
└── ulib/                   # The core library package
    ├── attack.py           # Defines base attack classes (UnivAttack, OptimAttack) and StopCriteria
    ├── attacks/            # Implementations of various UAP methods
    │   ├── df_uap.py       # Implementation of the DF_UAP attack
    │   ├── dt_uap.py       # Implementation of the DT_UAP attack
    │   ├── fff.py          # Implementation of the FFF attack
    │   └── ...
    ├── data/                   # Data-related classes
    │   ├── data_extractor.py   # Utility for splitting and filtering datasets
    │   └── subset_folder.py    # Utility for creating partial datasets from image folders
    ├── logger.py               # Defines the Logger class for experiment tracking
    ├── pert_module.py          # Defines the PertModule class for wrapping models and applying perturbations
    ├── activation_extractor.py # Utility for extracting activations from model layers
    ├── eval.py                 # Evaluation functions for analyzing perturbation performance
    └── utils.py                # General utility functions used throught the API
```

### Key Files and Directories:

*   **`ulib/`**: This directory contains the core library code.
*   **`ulib/attack.py`**: Defines the base classes for all attacks, including [`UnivAttack`](ulib/attack.py) and [`OptimAttack`](ulib/attack.py). It also includes the [`StopCriteria`](ulib/attack.py) class for defining stopping conditions.
*   **`ulib/attacks/`**: Contains implementations of various UAP attacks. Each file in this directory implements a specific attack algorithm.
*   **`ulib/pert_module.py`**: Defines the [`PertModule`](ulib/pert_module.py) class, which is used to wrap the target model and manage the adversarial perturbation.
*   **`ulib/logger.py`**: Defines the [`Logger`](ulib/logger.py) class, which is used for logging experiment metrics, hyperparameters, and generated adversarial examples.
*   **`ulib/data/data_extractor.py`**: Defines the [`DataExtractor`](ulib/data/data_extractor.py) class, which provides utilities for splitting and filtering datasets based on various criteria.
*   **`notebooks/`**: This directory contains Jupyter notebooks that demonstrate how to use the library. The notebooks cover various aspects of UAP generation, including loading datasets, training models, and evaluating attacks.

## Core Classes

### [`PertModule`](ulib/pert_module.py)

The [`PertModule`](ulib/pert_module.py) is a crucial component of the library. It wraps the original model and manages the perturbation.

*   **Initialization:** Takes the original model and perturbation parameters (e.g., epsilon) as input.
*   **Functionality:**
    *   Adds a perturbation to the input during the forward pass.
    *   Clips the perturbation to ensure it stays within the specified bounds.
    *   Allows enabling/disabling the perturbation.
    *   Provides methods for getting the perturbation (`get_pert`), initializing it randomly (`random_init`), and projecting it onto a valid space (`project`).
    *   The `to_image` method converts the perturbation to a visualizable image.

### [`UnivAttack`](ulib/attack.py)

The [`UnivAttack`](ulib/attack.py) class (and its subclass [`OptimAttack`](ulib/attack.py)) serves as the base class for all attack implementations.

*   **Initialization:** Takes the [`PertModule`](ulib/pert_module.py), optimizer (for `OptimAttack`), and other attack-specific parameters as input. It accepts optional `log_dir` argument to log metrics during training.
*   **Functionality:**
    *   `fit()`: The main method for generating the UAP. It iterates through the training dataset and updates the perturbation. 
    *   `close()`: Cleans up resources after the attack is finished.
    *   Handles logging and checkpointing.

## Usage Examples

### Basic Attack Example (GD\_UAP)

This example demonstrates how to use the `GD_UAP` attack to generate a universal adversarial perturbation.

```python
import torch
import torch.optim as optim
from ulib.pert_module import PertModule
from ulib.attacks.gd_uap import GD_UAP
from ulib.attack import StopCriteria
from notebooks.experiment_robust import load_robust_experiment

# 1. Load a model and data loaders
model, dl_train, dl_eval = load_robust_experiment("Standard", "cifar10")

# 2. Create a PertModule
pert_model = PertModule(model, data_shape=next(iter(dl_train))[0].shape[1:], eps=8/255)

# 3. Define an optimizer
optimizer = optim.Adam(pert_model.parameters(), lr=1e-3)

# 4. Initialize the attack
attack = GD_UAP(
    pert_model=pert_model,
    optimizer=optimizer,
    data_dependant=True,
    sat_thresh=0.5,
    sat_delta=1e-5,
    log_dir="logs",
)

# 5. Define stopping criteria
stop = StopCriteria(max_epochs=10, max_time=60 * 10)

# 6. Run the attack
pert_tensor = attack.fit(dl_train, dl_eval, stop)

# 7. Close the attack
attack.close()

print("UAP Generated!")
```

### Advanced Attack Parameters (USGD)

The following code snippet demonstrates some of the advanced features supported by `OptimAttack` and `UniversalAttack`.
For a full review of all supported features check the documentation of these classes. 

```python
import torch
import torch.optim as optim
from ulib import eval
from ulib.pert_module import PertModule
from ulib.attacks.usgd import USGD
from ulib.attack import StopCriteria
from notebooks.experiment_robust import load_robust_experiment

# 1. Load a model and data loaders
model, dl_train, dl_eval = load_robust_experiment("Standard", "cifar10")

# 2. Create a PertModule
pert_model = PertModule(model, 
    data_shape=next(iter(dl_train))[0].shape[1:], 
    eps=16/255, 
    norm=float("inf"), 
    random_init=False,
)

# 3. Define an optimizer
optimizer = optim.Adam(pert_model.parameters(), lr=1e-3)

# 4. Choose loss function
criterion = torch.nn.CrossEntropyLoss()

# 5. (Optional) Choose LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

# 6. (Optional) Create a grad-scaler, which activates torch autocast. 
grad_scaler = torch.GradScaler(device=pert_model.device.type)

# 7. Initialize the attack
attack = USGD(
    # attack-specific params
    pert_model=pert_model,
    optimizer=optimizer,
    criterion=criterion,
    skip_already_fooled=False,
    # general params
    scheduler=scheduler,
    grad_scaler=grad_scaler,
    sched_on_batch=True,
    eval_freq=100,
    targeted=True,
    metric_func=eval.attack_success_ratio,
    log_dir="logs",
)

# 8. Define stopping criteria
stop = StopCriteria(max_epochs=10, max_time=60 * 10, patience=5, patience_delta=0.001)

# 9. Run the attack
pert_tensor = attack.fit(dl_train, dl_eval, stop)

# 10. Close the attack
attack.close()

print("UAP Generated!")
```

### Exploring the Notebooks

The notebooks directory contains several Jupyter notebooks that demonstrate how to use the library. These notebooks provide practical examples of how to generate UAPs using different attack methods and evaluate their effectiveness.  Refer to the list in the Repository Structure section for a description of each notebook.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

*   Gilad Freidkin
*   Guy Cohen
