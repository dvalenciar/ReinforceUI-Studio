<p align="center"> <a href="https://docs.reinforceui-studio.com/welcome"><img src="https://raw.githubusercontent.com/dvalenciar/ReinforceUI-Studio/main/media_resources/cover_RL.png" alt="ReinforceUI" width="100%"> </a> </p> 
<h1 align="center">🚀 ReinforceUI Studio: Reinforcement Learning Made Simple</h1> 
<p align="center"> Intuitive, Powerful, and Hassle-Free RL Training & Monitoring – All in One Place. </p> 
<p align="center"> <a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions"> <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/pytest.yml?label=CI&branch=main" alt="Build Status"> </a> 
<a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions"> <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/docker-publish.yml?label=Docker&branch=main" alt="Docker Status"> </a> 
<a href="https://github.com/dvalenciar/ReinforceUI-Studio/actions/workflows/formatting.yml"> <img src="https://img.shields.io/github/actions/workflow/status/dvalenciar/ReinforceUI-Studio/formatting.yml?label=Formatting&branch=main" alt="Formatting Status"> </a> 
<a href="https://docs.reinforceui-studio.com/"> <img src="https://img.shields.io/badge/Docs-Up-blue" alt="Documentation"> </a> <a href="https://opensource.org/licenses/MIT"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"> </a> 
<a href="https://www.python.org/downloads/release/python-310/"> <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python Version"> </a> </p>

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
The recommended way to use ReinforceUI Studio is by running it locally.
This gives you full control over the application, with no extra tools needed.

Follow these simple steps:

1. Clone the repository and install dependencies

```bash
git clone https://github.com/dvalenciar/ReinforceUI-Studio.git
cd ReinforceUI-Studio
pip install -r requirements.txt
```

2. Run the application

```bash
python main.py 
```

That's it! You’re ready to start training and monitoring your Reinforcement Learning agents through an intuitive GUI.

✅ Tip:
If you encounter any issues, check out the [Installation Guide](https://docs.reinforceui-studio.com/user_guides/installation) in the full documentation.

## Why you should use ReinforceUI Studio
* 🚀 Instant RL Training: Configure environments, select algorithms, set hyperparameters – all in seconds.
* 🖥️ Real-Time Dashboard: Watch your agents learn with live performance curves and metrics.
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


### Acknowledgements
This project was inspired by the CARES Reinforcement Learning Package from the University of Auckland 
