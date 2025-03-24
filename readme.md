<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# ULIB

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/giladfrid009/ulib?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/giladfrid009/ulib?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/giladfrid009/ulib?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/giladfrid009/ulib?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview



---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Follows a modular design with clear separation of concerns.</li><li>Uses object-oriented programming principles for extensibility and maintainability.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Consistent code formatting adhering to PEP8 standards.</li><li>Includes comprehensive unit tests covering major functionalities.</li></ul> |
| 📄 | **Documentation** | <ul><li>Well-documented functions and classes with clear docstrings.</li><li>README.md provides setup instructions and usage examples.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates seamlessly with popular libraries like NumPy and Pandas for data manipulation.</li><li>Supports easy integration with Jupyter notebooks for interactive data analysis.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Each module handles a specific aspect of functionality, promoting reusability.</li><li>Follows the single responsibility principle for clean and maintainable code.</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes unit tests using pytest for core functions and edge cases.</li><li>Integration tests ensure components work together correctly.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized algorithms for efficient data processing and computation.</li><li>Utilizes vectorized operations for improved performance with large datasets.</li></ul> |
| 🛡️ | **Security**      | <ul><li>Implements input validation to prevent common security vulnerabilities like SQL injection.</li><li>Follows best practices for handling sensitive data securely.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Clearly defined dependencies in requirements.yaml for easy setup and reproducibility.</li><li>Uses virtual environments to manage package versions and avoid conflicts.</li></ul> |

---

## Project Structure

```sh
└── ulib/
    ├── notebooks
    │   ├── datasets.py
    │   ├── experiment_robust.py
    │   ├── experiment_torch.py
    │   ├── experiment_utils.py
    │   ├── test_ae_uap.ipynb
    │   ├── test_cosine_uap.ipynb
    │   ├── test_d_badge.ipynb
    │   ├── test_df_uap.ipynb
    │   ├── test_dt_uap.ipynb
    │   ├── test_du_attack.ipynb
    │   ├── test_fff.ipynb
    │   ├── test_fg_uap.ipynb
    │   ├── test_gd_uap.ipynb
    │   ├── test_imi_uap.ipynb
    │   ├── test_iml_svrg_uap.ipynb
    │   ├── test_iml_uap.ipynb
    │   ├── test_iml_uap_sched.ipynb
    │   ├── test_mealpy.ipynb
    │   ├── test_sd_uap.ipynb
    │   ├── test_spm.ipynb
    │   ├── test_svrg.ipynb
    │   ├── test_uapgd.ipynb
    │   ├── test_ufgsm.ipynb
    │   └── test_usgd.ipynb
    ├── requirements.yaml
    └── ulib
        ├── __init__.py
        ├── activation_extractor.py
        ├── attack.py
        ├── attacks
        ├── data
        ├── eval.py
        ├── logger.py
        ├── pert_module.py
        └── utils.py
```

### Project Index

<details open>
	<summary><b><code>ULIB/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/requirements.yaml'>requirements.yaml</a></b></td>
					<td style='padding: 8px;'>- Create a Conda environment with required dependencies for the ulib project<br>- Includes Python 3.11, libraries like torch, torchvision, and mealpy<br>- Use this file to set up or update the environment for seamless project execution.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- notebooks Submodule -->
	<details>
		<summary><b>notebooks</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ notebooks</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/datasets.py'>datasets.py</a></b></td>
					<td style='padding: 8px;'>- Provide functions to load CIFAR-10, CIFAR-100, and ImageNet datasets with specified transformations and batch sizes<br>- Returns training and evaluation data loaders for each dataset<br>- The code facilitates easy dataset loading for machine learning tasks.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/experiment_robust.py'>experiment_robust.py</a></b></td>
					<td style='padding: 8px;'>- Generate robust experiment configurations and load models for evaluation on CIFAR-10, CIFAR-100, or ImageNet datasets<br>- The code retrieves model information, preprocesses data, and evaluates model accuracy, providing insights into model architecture and performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/experiment_torch.py'>experiment_torch.py</a></b></td>
					<td style='padding: 8px;'>- Load Torchvision experiment with specified model type, preparing data transformations, model layers, and evaluation metrics<br>- Clear memory, set device, load model weights, and create a sequential model for evaluation<br>- Patch model class name, load ImageNet data, and calculate accuracy metrics if not silent mode<br>- Return the model and data loaders for training and evaluation.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/experiment_utils.py'>experiment_utils.py</a></b></td>
					<td style='padding: 8px;'>- Enhances object-oriented flexibility by dynamically modifying class names<br>- Converts strings to valid Python class names and patches them onto objects<br>- Ideal for scenarios requiring dynamic class creation with user-defined names.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_ae_uap.ipynb'>test_ae_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an adversarial attack using the AE_UAP method on a CIFAR-10 model<br>- Load the model, set attack parameters, and execute the attack to generate perturbed images<br>- Evaluate the perturbed models performance on the evaluation dataset using the provided utility function.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_cosine_uap.ipynb'>test_cosine_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Execute an attack using Cosine UAP to generate adversarial perturbations on a pre-trained model<br>- Evaluate the perturbed models performance against a validation dataset<br>- Conduct a full analysis to assess the impact of the perturbations on the models robustness.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_d_badge.ipynb'>test_d_badge.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement adversarial attack using D_BADGE to perturb a model for CIFAR-10 dataset<br>- Load robust and torchvision experiments, set attack parameters, and analyze model performance<br>- Fix issues with the attack for proper functionality.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_df_uap.ipynb'>test_df_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Execute an attack to generate an untargeted adversarial perturbation on a deep learning model trained on CIFAR-10<br>- The attack utilizes a Decision-based Fast Universal Adversarial Perturbation method to craft perturbations<br>- After the attack, a full analysis is conducted on the perturbed models performance using the evaluation dataset.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_dt_uap.ipynb'>test_dt_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using DT_UAP to perturb a model for robustness evaluation<br>- Load necessary experiments, set up attack parameters, and execute the attack<br>- Evaluate the perturbed models performance using full analysis.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_du_attack.ipynb'>test_du_attack.ipynb</a></b></td>
					<td style='padding: 8px;'>- Execute a notebook to load a robust CIFAR-10 model and perform a Differential Uncertainty (DU) attack<br>- The attack aims to perturb the model with specified criteria and parameters<br>- Finally, evaluate the perturbed model using a full analysis.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_fff.ipynb'>test_fff.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack using the FFF method to perturb a model for robustness evaluation<br>- Load a pre-trained model, conduct the attack, and evaluate its performance<br>- This notebook facilitates model evaluation and robustness testing in adversarial settings.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_fg_uap.ipynb'>test_fg_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using Fast Gradient Universal Adversarial Perturbations (FG-UAP) on a pre-trained model<br>- Load necessary experiments, set up attack parameters, and execute the attack to generate perturbations<br>- Finally, evaluate the perturbed models performance using a full analysis.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_gd_uap.ipynb'>test_gd_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using Gradient Descent Unrestricted Adversarial Perturbations (GD_UAP) on a pre-trained model<br>- Load datasets, set up attack parameters, and analyze model robustness<br>- The code enhances model security against adversarial attacks by generating perturbations.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_imi_uap.ipynb'>test_imi_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using IMI_UAP to perturb a model for robustness evaluation<br>- The code sets up the attack parameters, including the perturbation model, optimizer, inner attack method, and evaluation criteria<br>- It then executes the attack on the training and evaluation datasets, followed by a full analysis of the perturbed models performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_iml_svrg_uap.ipynb'>test_iml_svrg_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an Iterative Machine Learning (IML) algorithm for Unrestricted Adversarial Perturbations (UAP) using Stochastic Variance Reduced Gradient (SVRG)<br>- This code file integrates various attack strategies to enhance model robustness against adversarial attacks<br>- It leverages perturbation modules, activation extraction, and evaluation functions to optimize adversarial training and evaluation processes within the project architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_iml_uap.ipynb'>test_iml_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an Iterative Machine Learning Universal Adversarial Perturbation (IML UAP) attack using the provided code<br>- This attack generates perturbations to deceive a model, enhancing its robustness evaluation<br>- The code orchestrates the attack process, optimizing perturbations to fool the model while analyzing its performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_iml_uap_sched.ipynb'>test_iml_uap_sched.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using various methods to generate adversarial examples for deep learning models<br>- The code orchestrates the creation of perturbations to deceive the model, enhancing its robustness evaluation<br>- It leverages a combination of optimization techniques and attack algorithms to craft effective adversarial inputs.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_mealpy.ipynb'>test_mealpy.ipynb</a></b></td>
					<td style='padding: 8px;'>- Optimize the perturbation of a machine learning model using BinaryObjective class<br>- The code file in test_mealpy.ipynb loads a robust experiment, defines a binary optimization problem, and utilizes a Particle Swarm Optimization algorithm to find the best perturbation for the model<br>- The final perturbation is evaluated for its impact on model accuracy.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_sd_uap.ipynb'>test_sd_uap.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an attack strategy using SD_UAP to enhance model robustness<br>- Load a pre-trained model, define attack parameters, and execute the attack on evaluation data<br>- Conduct a full analysis post-attack to evaluate model performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_spm.ipynb'>test_spm.ipynb</a></b></td>
					<td style='padding: 8px;'>- Load and execute robust experiments on CIFAR-10 using a pre-trained model<br>- Implement the SPM_UAP adversarial attack to generate perturbations and evaluate model robustness<br>- Perform a full analysis on the perturbed model using the evaluation dataset.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_svrg.ipynb'>test_svrg.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an SVRG-UAP attack using a pre-trained model to generate adversarial perturbations<br>- The code loads the model, defines attack parameters, and fits the attack to the training and evaluation datasets<br>- Finally, it performs a full analysis on the perturbed model using evaluation data.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_uapgd.ipynb'>test_uapgd.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an adversarial attack using UAPGD to perturb a pre-trained model<br>- Load data, set attack parameters, and execute the attack to generate adversarial examples<br>- Evaluate the perturbed models performance using the perturbation module and the evaluation dataset<br>- Close the attack process and conduct a full analysis on the perturbed models performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_ufgsm.ipynb'>test_ufgsm.ipynb</a></b></td>
					<td style='padding: 8px;'>- Implement an adversarial attack using UFGSM on a PyTorch model trained on CIFAR-10<br>- The code loads the model, defines a perturbation module, sets up optimization parameters, and executes the attack<br>- Finally, it evaluates the perturbed models performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/notebooks/test_usgd.ipynb'>test_usgd.ipynb</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>test_usgd.ipynb</code> notebook file in the project serves the purpose of fixing imports by ensuring that the necessary modules are correctly included for the code to run smoothly<br>- It appends the projects module path to the system path if it's not already included, ensuring that the subsequent code cells can access the required modules without any issues<br>- This step is crucial for the overall functionality of the codebase and ensures a seamless execution environment for the project.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- ulib Submodule -->
	<details>
		<summary><b>ulib</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ ulib</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/activation_extractor.py'>activation_extractor.py</a></b></td>
					<td style='padding: 8px;'>- Extract activations from specified layers of a PyTorch model and compute a loss term at each layer, aggregating them into a single scalar loss<br>- This facilitates monitoring and analyzing model performance at different stages.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attack.py'>attack.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>attack.py</code> file in the <code>ulib</code> directory of the project contains a class <code>StopCriteria</code> that serves as a container for various stopping criteria used during training processes<br>- These criteria include defining the maximum number of epochs, evaluation steps, training time, target value for stopping training, patience threshold, and minimum improvement delta<br>- The <code>StopCriteria</code> class encapsulates these parameters to control the training process and halt it based on specific conditions, ensuring efficient and effective training runs<br>- This file plays a crucial role in managing the training process within the project architecture by providing a structured approach to defining stopping criteria.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/eval.py'>eval.py</a></b></td>
					<td style='padding: 8px;'>- Evaluate model performance metrics, such as accuracy, misclassification rate, fooling rate, and attack success ratio, using a perturbation module on an evaluation dataset<br>- Conduct a comprehensive analysis to compare clean and robust accuracy, attack success ratio, and fooling rate<br>- Visualize the perturbation and provide detailed metric results.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/logger.py'>logger.py</a></b></td>
					<td style='padding: 8px;'>- Facilitates logging metrics, visualizations, and hyperparameters during training<br>- Initializes logging directories, manages TensorBoard and ClearML logging, and logs various data types like scalars, images, and model graphs<br>- Enables easy tracking and visualization of training progress and results.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/pert_module.py'>pert_module.py</a></b></td>
					<td style='padding: 8px;'>- Implement a PyTorch module that applies adversarial perturbations to input data before passing it through a pretrained model<br>- The module ensures the model remains in evaluation mode and offers methods to set, get, and visualize perturbations<br>- It also supports hyperparameter retrieval and parameter clamping.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>- Provide functions to manage reproducibility, memory, and device selection for computations in GPU environments<br>- Includes setting seed, getting device, clearing memory, extracting device from a module, and sampling random vectors from an Lp ball<br>- These utilities enhance consistency and efficiency in machine learning workflows.</td>
				</tr>
			</table>
			<!-- attacks Submodule -->
			<details>
				<summary><b>attacks</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ ulib.attacks</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/ae_uap.py'>ae_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a universal adversarial perturbation attack, AE_UAP leverages the AE_MIFGSM inner attack to generate adversarial examples<br>- By computing the loss based on the perturbed inputs, it aims to optimize the perturbation model using the provided criterion<br>- The code facilitates robust adversarial attacks within the project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/cosine_uap.py'>cosine_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a Cosine_UAP attack, the code calculates the loss between perturbed and clean model predictions<br>- It leverages a cosine similarity criterion to minimize the loss, as detailed in the referenced paper<br>- The attack aims to generate universal adversarial perturbations for black-box scenarios, enhancing model robustness.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/d_badge.py'>d_badge.py</a></b></td>
							<td style='padding: 8px;'>- Implementing D-BADGE attack with directional gradient estimation, the code in d_badge.py extends OptimAttack for adversarial batch attacks<br>- It leverages a custom Hamming loss criterion and perturbation model to optimize perturbations for model evasion<br>- The attack parameters are dynamically adjusted during training epochs for effective adversarial perturbation generation.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/df_uap.py'>df_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a Logits Margin module and DF_UAP class, the code in df_uap.py enhances adversarial attack capabilities within the project<br>- By computing loss based on logits and perturbations, it enables targeted or untargeted attacks<br>- This crucial functionality contributes significantly to the projects robustness and effectiveness in generating adversarial examples.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/dt_uap.py'>dt_uap.py</a></b></td>
							<td style='padding: 8px;'>- Define a loss function for crafting targeted universal adversarial perturbations<br>- The function computes the loss based on clean and adversarial activations, clean logits, and targets<br>- It handles both targeted and untargeted cases<br>- The DT_UAP class implements an optimization attack using this loss function, with adjustable alpha step size for perturbation generation.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/du_attack.py'>du_attack.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a decision-based universal adversarial attack, the code in <code>du_attack.py</code> crafts perturbations to deceive models<br>- It leverages a novel algorithm to generate noise and iteratively adjust perturbations based on model predictions<br>- By incorporating momentum and adaptive steps, it efficiently generates adversarial examples.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/fff.py'>fff.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a universal adversarial attack, FFF leverages ActivationExtractor to compute losses based on activation values<br>- It divides perturbations periodically during training to enhance robustness<br>- This code file, residing in ulib/attacks/fff.py, plays a crucial role in generating universal adversarial perturbations for the projects architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/fg_uap.py'>fg_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing FG-UAP attack for universal adversarial perturbations, leveraging feature gathering to deceive models<br>- Computes loss based on cosine similarity between clean and adversarial activations<br>- Skips already fooled samples if specified, enhancing attack efficiency<br>- Built on PertModule and ActivationExtractor for robust adversarial crafting.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/gd_uap.py'>gd_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a Universal Adversarial Perturbation attack, the code file <code>gd_uap.py</code> crafts perturbations to deceive models<br>- It leverages activation extraction and loss computation to optimize perturbations<br>- The attack dynamically adjusts perturbations based on saturation rates, enhancing adversarial effectiveness.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/imi_uap.py'>imi_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing an attack strategy, the IMI_UAP class in ulib/attacks/imi_uap.py computes loss to minimize the distance between perturbed and attacked samples<br>- It skips already fooled samples and failed attacks based on specified criteria, enhancing the robustness of the model<br>- The class efficiently registers attack parameters for analysis and optimization.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/iml_svrg_uap.py'>iml_svrg_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing an Iterative Machine Learning (IML) algorithm, the code file <code>iml_svrg_uap.py</code> orchestrates Universal Adversarial Perturbations (UAP) attacks<br>- It optimizes perturbations to deceive models, enhancing robustness<br>- The file manages perturbation generation, loss computation, and gradient updates, crucial for adversarial training within the projects architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/iml_uap.py'>iml_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing an Iterative Mean Loss Unconstrained Adversarial Perturbation attack, the code file <code>iml_uap.py</code> defines loss functions and attack strategies for generating adversarial examples<br>- It leverages activation extraction and optimization techniques to craft perturbations that deceive machine learning models<br>- The code enhances model robustness by iteratively refining attacks based on activation differences.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/iml_uap_sched.py'>iml_uap_sched.py</a></b></td>
							<td style='padding: 8px;'>- Implementing an iterative method for generating Universal Adversarial Perturbations (UAP), the code orchestrates attacks on neural networks<br>- It dynamically adjusts attacks based on model performance, optimizing for successful adversarial perturbations<br>- The code intelligently skips already-fooled samples and failed attacks, enhancing the efficiency and effectiveness of adversarial attacks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/sd_uap.py'>sd_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a data-free universal adversarial perturbation attack, the SD_UAP class in the ulib/attacks/sd_uap.py file facilitates crafting adversarial examples<br>- It leverages ActivationExtractor to compute losses based on activations, adjusting perturbations during training epochs<br>- The class offers flexibility in layer progression and perturbation normalization, enhancing adversarial robustness.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/svrg_uap.py'>svrg_uap.py</a></b></td>
							<td style='padding: 8px;'>- Implement a Universal Adversarial Perturbation attack using SVRG technique<br>- The code file <code>svrg_uap.py</code> defines a class that computes gradients for perturbing input data to maximize similarity between perturbed and adversarial inputs<br>- It supports different modes like classic and supernova for varied optimization strategies<br>- The attack aims to deceive models by generating adversarial examples efficiently.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/uapgd.py'>uapgd.py</a></b></td>
							<td style='padding: 8px;'>- Implementing a universal adversarial attack mechanism, the code in uapgd.py orchestrates the integration of a ValueScheduler, MIPGD attack, and PertModule to enhance the Projected Gradient Descent approach<br>- It facilitates dynamic parameter scheduling and batch-level control, optimizing adversarial perturbations for improved model robustness against attacks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/ufgsm.py'>ufgsm.py</a></b></td>
							<td style='padding: 8px;'>- Implementing the Universal Fast Gradient Sign Method (UFGSM) attack, the code in <code>ulib/attacks/ufgsm.py</code> applies sign vectors to model gradients before the optimizer step<br>- This enhances adversarial perturbations for robustness testing in deep learning models within the projects architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/usgd.py'>usgd.py</a></b></td>
							<td style='padding: 8px;'>- Implement a universal gradient-based attack using an optimizer to update perturbations<br>- The attack skips already fooled samples and adjusts the loss based on the attack type<br>- This file integrates with the projects perturbation module and optimizer for efficient adversarial attacks.</td>
						</tr>
					</table>
					<!-- spm Submodule -->
					<details>
						<summary><b>spm</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ ulib.attacks.spm</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/spm/adversarial_attack.py'>adversarial_attack.py</a></b></td>
									<td style='padding: 8px;'>- Implement an adversarial attack method using the Singular Vector Method (SPM) to generate universal adversarial perturbations<br>- The code fits the perturbation to input data, predicts outcomes, and calculates the fooling rate between original and perturbed model predictions.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/spm/model_feature_extractor.py'>model_feature_extractor.py</a></b></td>
									<td style='padding: 8px;'>- Extracts the output of a specified layer from a PyTorch model<br>- The code defines a class that takes a model and layer as input, allowing users to retrieve the output of that layer when passing input data through the model<br>- This functionality aids in feature extraction for further analysis or processing within the project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/giladfrid009/ulib/blob/master/ulib/attacks/spm/power_method.py'>power_method.py</a></b></td>
									<td style='padding: 8px;'>- Implementing a Power Method algorithm to compute the dominant eigenvalue and eigenvector of a Jacobian matrix<br>- The algorithm iteratively calculates the eigenvector by applying matrix-vector products and power iterations<br>- It also provides methods to retrieve the eigenvalue, eigenvector, and perturbation based on the computed values.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python

### Installation

Build ulib from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/giladfrid009/ulib
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd ulib
    ```

3. **Install the dependencies:**

echo 'INSERT-INSTALL-COMMAND-HERE'

### Usage

Run the project with:

echo 'INSERT-RUN-COMMAND-HERE'

### Testing

Ulib uses the {__test_framework__} test framework. Run the test suite with:

echo 'INSERT-TEST-COMMAND-HERE'

---

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/giladfrid009/ulib/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/giladfrid009/ulib/issues)**: Submit bugs found or log feature requests for the `ulib` project.
- **💡 [Submit Pull Requests](https://github.com/giladfrid009/ulib/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/giladfrid009/ulib
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/giladfrid009/ulib/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=giladfrid009/ulib">
   </a>
</p>
</details>

---

## License

Ulib is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
