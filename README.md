# Universal Adversarial Perturbations Library (ULib)

ULib is a modular, PyTorch-based framework for generating universal adversarial perturbations (UAPs) that can fool deep neural networks. The library comes with a variety of attack implementations, tools for dataset manipulation, and detailed experiment logging—making it an ideal solution for both research and practical evaluation of adversarial attacks.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Attack Example (GD_UAP)](#basic-attack-example-gd_uap)
  - [Advanced Attack Parameters (USGD)](#advanced-attack-parameters-usgd)
- [Repository Structure](#repository-structure)
- [API Reference](#api-reference)
  - [Core Classes](#core-classes)
    - [PertModule](#pertmodule)
    - [UnivAttack & OptimAttack](#univattack--optimattack)
    - [ActivationExtractor](#activationextractor)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## Features

- **Modular Attack Implementations:**  
  Implements various UAP generation methods such as GD-UAP, DF-UAP, DT-UAP, FFF, UAPGD, FG-UAP, SD-UAP, Cosine-UAP, AE-UAP, D-BADGE, and more.

- **Flexible Perturbation Module:**  
  The [`PertModule`](ulib/pert_module.py) encapsulates a target model and manages adversarial perturbations seamlessly.

- **Extensive Experiment Notebooks:**  
  A set of Jupyter notebooks demonstrates different attack strategies on various datasets and models.

- **Built-in Logging:**  
  Integrated logging via the [`Logger`](ulib/logger.py) class tracks metrics, hyperparameters, and generated adversarial examples throughout experiments.

- **Customizable Stopping Criteria:**  
  Flexible stopping conditions (e.g., time limits, epoch limits, early stopping) ensure controlled experiment runs.

- **Data Extraction Tools:**  
  Utilities for splitting datasets by correctness, class label, or confidence, and for creating partial datasets from image folders.

---

## Installation

To install the dependencies of this project, please use the ***uv*** dependency manager. (highly recommended)   
*Note:* If you dont have ***uv*** installed, see docs at the [official website](https://docs.astral.sh/uv/getting-started/installation/).

#### Dependencies:

The dependencies are located in two files:
* ```pyproject.toml``` - contains the general package names and versions, should be enough for standard installations.
* ```uv.lock``` - contains exact versions of all installed packages. 

#### Steps:

1. Clone Repo from github
2. Navigate to the repo folder
3. Run the following command in your terminal: ```uv sync```

#### Env Activation:

1. Navigate to the repo folder
2. Run the following command in your terminal: ```source ./.venv/bin/activate```

---

## Quick Start

Here is a simple code snippet demonstraing running a UAP attack. The attack here is Cosine-UAP, but for other attacks the procedure is similar. 

```python
import torch
from ulib import PertModule, StopCriteria
from ulib.attacks.cosine_uap import Cosine_UAP
from notebooks.experiment_robust import load_robust_experiment

# Load model and data loaders
model, dl_train, dl_eval = load_robust_experiment("Standard", "cifar10")

# Wrap the model with PertModule (epsilon = 8/255, L-infinity norm)
pert_model = PertModule(model, data_shape=(3, 32, 32), eps=8/255)

# Set up optimizer
optimizer = torch.optim.Adam(pert_model.parameters(), lr=1e-3)

# Initialize the attack
attack = Cosine_UAP(pert_model=pert_model, optimizer=optimizer)

# Define stopping criteria (max 10 epochs or 10 minutes)
stop = StopCriteria(max_epochs=10, max_time=600)

# Generate the UAP
pert_tensor = attack.fit(dl_train, dl_eval, stop)

print("UAP Generated!")
```

ULib is designed to be plug-and-play. Here’s how to generate a universal adversarial perturbation on a sample dataset:

1. **Load a pretrained model and data loaders.**
2. **Wrap the model with a `PertModule` to manage the perturbation.**
3. **Initialize an attack with the desired hyperparameters.**
4. **Run the attack with specified stopping criteria.**

Refer to the [Usage Examples](#usage-examples) section for complete code snippets.

---

## Usage Examples

### Basic Attack Example (GD_UAP)

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
    logging_enable=True,
)

# 5. Define stopping criteria
stop = StopCriteria(max_epochs=10, max_time=60 * 10)

# 6. Run the attack
pert_tensor = attack.fit(dl_train, dl_eval, stop)

print("UAP Generated!")
```

### Advanced Attack Parameters (USGD)

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

# 2. Create a PertModule with advanced settings
pert_model = PertModule(
    model, 
    data_shape=next(iter(dl_train))[0].shape[1:], 
    eps=16/255, 
    norm=float("inf"), 
    random_init=False,
)

# 3. Define an optimizer
optimizer = optim.Adam(pert_model.parameters(), lr=1e-3)

# 4. Choose loss function
criterion = torch.nn.CrossEntropyLoss()

# 5. (Optional) Chooe LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

# 7. Initialize the attack with extended settings
attack = USGD(
    # attack-specific params
    pert_model=pert_model,
    optimizer=optimizer,
    criterion=criterion,
    skip_already_fooled=False,
    # general params
    scheduler=scheduler,
    sched_on_batch=True,
    mixed_precision=True,
    eval_freq=0.5,
    targeted=False,
)

# 8. Define stopping criteria with patience
stop = StopCriteria(max_epochs=10, max_time=60 * 10, patience=5, patience_delta=0.001)

# 9. Run the attack
pert_tensor = attack.fit(dl_train, dl_eval, stop)


print("UAP Generated!")
```

---

## Repository Structure

```plaintext
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.yaml         # Environment/dependency configuration
├── notebooks/                # Jupyter notebooks demonstrating library usage
│   ├── datasets.py           # Utility functions for loading datasets
│   ├── experiment_robust.py  # Robust training experiment setup
│   ├── experiment_torch.py   # Standard PyTorch experiments
│   ├── experiment_utils.py   # Experiment utility functions
│   ├── test_df_uap.ipynb     # DF_UAP attack demonstration
│   ├── test_dt_uap.ipynb     # DT_UAP attack demonstration
│   ├── test_fff.ipynb        # FFF attack demonstration
│   └── ...
└── ulib/                     # Core library package
    ├── attack.py             # Base attack classes and StopCriteria
    ├── attacks/              # Implementations of UAP methods (e.g., df_uap.py, dt_uap.py, fff.py)
    ├── data/                 
    │   ├── data_extractor.py # Tools for splitting/filtering datasets
    │   └── tensor_loader.py  # Efficient tensor batch loader
    ├── logger.py             # Experiment logging utilities
    ├── pert_module.py        # PertModule for model perturbations
    ├── activation_extractor.py # Extract activations from model layers
    └── utils.py              # General utility functions
```

---

## API Reference

### Core Classes

#### PertModule

The [`PertModule`](ulib/pert_module.py) is responsible for applying and managing adversarial perturbations on inputs before they are passed to a pretrained model. Key features include:

- **Initialization:** Accepts the target model, input shape, epsilon (perturbation bound), norm constraint, and other parameters.
- **Perturbation Management:** Methods such as `random_init()`, `project()`, `get_pert()`, and `set_pert()` ensure that the perturbation remains within valid bounds.
- **Visualization:** The `to_image()` method converts the perturbation into a normalized image for easy visualization.

#### UnivAttack & OptimAttack

The [`UnivAttack`](ulib/attack.py) serves as the base class for all adversarial attacks. It handles:

- **Attack Lifecycle:**  
  Methods for training (`fit()`), evaluation (`evaluate()`), and checkpointing.
- **Logging & Metrics:**  
  Integration with the [`Logger`](ulib/logger.py) to track metrics and hyperparameters.

The subclass [`OptimAttack`](ulib/attack.py) extends this functionality by incorporating an optimizer, learning rate scheduler, and (optionally) mixed precision tools such as `grad_scaler` and `autocast` for efficient training.

#### ActivationExtractor

Located in [`ulib/activation_extractor.py`](ulib/activation_extractor.py), the `ActivationExtractor` allows you to capture intermediate activations from specified layers. This can be useful for analyzing model behavior or designing custom loss functions that depend on internal representations.

For a detailed review of all parameters and usage examples, please refer to the inline documentation in the source code.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository:** Create a personal fork and work on your feature branch.
2. **Coding Style:** Follow PEP 8 and ensure your code is well documented.
3. **Testing:** We would love if you include tests for new features or bug fixes.
4. **Pull Request:** Submit a pull request with a clear description of your changes.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Gilad Freidkin**
- **Guy Cohen**

---

Feel free to explore the repository and the provided notebooks to learn more about generating and evaluating universal adversarial perturbations with ULib!
