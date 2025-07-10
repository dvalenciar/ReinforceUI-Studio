<p align="center">
  <a href="https://docs.reinforceui-studio.com/welcome">
    <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/cover_RL.png" alt="ReinforceUI" width="100%">
  </a>
</p>

<h1 align="center"> Reinforcement Learning Made Simple</h1>

<p align="center">
  Intuitive, Powerful, and Hassle-Free RL Training & Monitoring – All in One Place.
</p>

<p align="center">
  <a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions/workflows/pytest.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/pytest.yml?style=for-the-badge&logo=github&label=CI" alt="Build Status">
  </a>

  <a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions/workflows/formatting.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/formatting.yml?style=for-the-badge&label=Formatting&branch=main" alt="Formatting Status">
  </a>

  <a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions/workflows/docker-publish.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/docker-publish.yml?style=for-the-badge&logo=docker&label=Docker" alt="Docker Status">
  </a>

  <a href="https://pypi.org/project/reinforceui-studio/">
    <img src="https://img.shields.io/pypi/v/reinforceui-studio?style=for-the-badge&logo=pypi&color=#44cc11" alt="PyPI version">
  </a>

  <a href="https://docs.reinforceui-studio.com/">
    <img src="https://img.shields.io/website?url=https%3A%2F%2Fdocs.reinforceui-studio.com&up_message=online&down_color=red&style=for-the-badge&label=Docs" alt="Documentation">
  </a>
</p>

<p align="center">
   <img src="https://img.shields.io/badge/python-3.10--3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python Version">
   <img src="https://img.shields.io/badge/Ubuntu-22--24-E95420?style=for-the-badge&logo=ubuntu&logoColor=white&&color=blue" alt="Ubuntu Version">
   <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License">
 
</p>


<p align="center">
   <img src="https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue" alt="Mlflow">
   <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="Mlflow">
</p>


<p align="center">
   <a href="https://pepy.tech/projects/reinforceui-studio">
    <img src="https://img.shields.io/pepy/dt/reinforceui-studio?style=for-the-badge&logo=pypi&color=blue" alt="PyPI Downloads">
   </a>
</p>

---
⭐️ If you find this project useful, please consider giving it a star! It really helps!

📚 Full Documentation: <a href="https://docs.reinforceui-studio.com" target="_blank">https://docs.reinforceui-studio.com</a>

🎬 Video Demo: [YouTube Tutorial](https://www.youtube.com/watch?v=itXyyttwZ1M)

---

## What is ReinforceUI Studio?

ReinforceUI Studio is a Python-based application designed to simplify Reinforcement Learning (RL) workflows through a beautiful, intuitive GUI.
No more memorizing commands, no more juggling extra repos – just train, monitor, and evaluate in a few clicks!

<p align="center"> <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/new_main_window_example.gif" width="80%"> </p>



## Quickstart
Getting started with ReinforceUI Studio is fast and easy!

### 🖥️ Install and Run Locally
The easiest way to use ReinforceUI Studio is by installing it directly from PyPI. This provides a hassle-free installation, allowing you to get started quickly with no extra configuration.

Follow these simple steps:

1. Install ReinforceUI Studio from PyPI

```bash
pip install reinforceui-studio
```

2. Run the application

```bash
reinforceui-studio
```

That's it! You’re ready to start training and monitoring your Reinforcement Learning agents through an intuitive GUI.

✅ Tip:
If you encounter any issues, check out the [Installation Guide](https://docs.reinforceui-studio.com/user_guides/installation) in the full documentation.

## Why you should use ReinforceUI Studio
* 🚀 Instant RL Training: Configure environments, select algorithms, set hyperparameters – all in seconds.
* 🖥️ Real-Time Dashboard: Watch your agents learn with live performance curves and metrics.
* 📊 Mlflow Integration: Automatically log and visualize your training runs with MLflow.
* 🧠 Multi-Algorithm Support: Train and compare multiple algorithms simultaneously.
* 📦 Full Logging: Automatically save models, plots, evaluations, videos, and training stats.
* 🔧 Easy Customization: Adjust hyperparameters or load optimized defaults.
* 🧩 Environment Support: Works with MuJoCo, OpenAI Gymnasium, and DeepMind Control Suite.
* 📊 Final Comparison Plots: Auto-generate publishable comparison graphs for your reports or papers.

## Quick Overview: Single and Multi-Algorithm Training

* **Single Training**: Choose an algorithm, tweak parameters, train & visualize.

* **Multi-Training**: Select several algorithms, run them simultaneously, and compare performances side-by-side.

<table align="center">
  <tr>
    <th>Selection Window</th>
    <th>Main Window Display</th>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/single_selection.png" width="400"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/single_selection_main_window.png" width="400"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/multiple_selection.png" width="400"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/multiple_selection_main_window.png" width="400"></td>
  </tr>
</table>

## Mlflow Integration
<table align="center">
  <tr>
    <th>Example of MLflow Dashboard</th>
    <th>Example of MLflow Metrics</th>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/mlflow_dashboard_1.png" width="400"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/mlflow_dashboard_2.png" width="400"></td>
  </tr>
  
</table>

## Supported Algorithms
ReinforceUI Studio supports the following algorithms:

| Algorithm | Description |
| --- | --- |
| **CTD4** | Continuous Distributional Actor-Critic Agent with a Kalman Fusion of Multiple Critics |
| **DDPG** | Deep Deterministic Policy Gradient |
| **DQN** | Deep Q-Network |
| **PPO** | Proximal Policy Optimization |
| **SAC** | Soft Actor-Critic |
| **TD3** | Twin Delayed Deep Deterministic Policy Gradient |
| **TQC** | Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics |



## Results Examples
Below are some examples of results generated by ReinforceUI Studio, showcasing the evaluation curves along with snapshots of the policies in action.

| **Algorithm** | **Platform** | **Environment**    | **Curve**                                                       | **Video**                                                                                        |
|---------------|--------------|--------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **SAC**       | DMCS         | Walker Walk        | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/SAC_walker_walk.png" width="200">        | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/walker_walk.gif" width="200">       | 
| **TD3**       | MuJoCo       | HalfCheetah v5     | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/TD3_HalfCheetah-v5.png" width="200">     | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/HalfCheetah.gif" width="200">       |
| **CDT4**      | DMCS         | Ball in cup catch  | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/CTD4_ball_in_cup_catch.png" width="200"> | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/ball_in_cup_catch.gif" width="200"> | 
| **DQN**       | Gymnasium    | CartPole v1        | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/DQN_CartPole-v1.png" width="200">        | <img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/result_examples/CartPole.gif" width="200">          | 


## Citation
If you find ReinforceUI Studio useful for your research or project, please kindly star this repo and cite is as follows:

```
@misc{reinforce_ui_studio_2025,
  title = { ReinforceUI Studio: Simplifying Reinforcement Learning Training and Monitoring},
  author = {David Valencia Redrovan},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dvalenciar/ReinforceUI-Studio.}
}
```


## Why Star ⭐ this Repository?
Your support helps the project grow!
If you like ReinforceUI Studio, please star ⭐ this repository and share it with friends, colleagues, and the RL community!
Together, we can make Reinforcement Learning accessible to everyone!

## License
ReinforceUI Studio is licensed under the MIT License. You are free to use, modify, and distribute this software, 
provided that the original copyright notice and license are included in any copies or substantial portions of the software.
