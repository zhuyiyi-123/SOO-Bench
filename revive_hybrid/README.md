# License

POLIXIR REVIVE is released subject to the [POLIXIR Commercial License](./License_en.txt), unless otherwise indicated. Please read and agree the accompanying license file carefully before downloading, installing or using the Polixir software or any accompanying files.

Unless otherwise noted, all files Copyright © 2021 - 2022 Polixir Technologies, Co., Ltd.

# Introduction

REVIVE is a general platform that aims to bring automatic decision-making to real-world scenarios.
The platform operates in a pipeline of two steps:
1. **Venv Training:** A virtual-environment model is trained from any offline data to mimic each agent's policy along with the transition between states (also known as nature's policy).
2. **Policy Training:** Treat one of the agents as the active agent and freeze others as its environment. Train the active agent with reinforcement learning to derive a better policy for the agent.


# Documentation

The tutorials and API documentation are hosted on https://revive.cn/help/polixir-revive-sdk/index.html.

# Installation

### Requirements

-   Linux x86\_64
-   [Python](https://python.org/): v3.7.0+ / v3.8.0+ / v3.9.0+
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (If NVIDIA
    CUDA GPU device is available.)

###  Install REVIVE

You can install the latest sversion of the from a cloned Git repository:

    $ git clone https://agit.ai/Polixir/revive
    $ cd revive
    $ pip install -e .

You can also view and download all versions of the code package through the releases page.

**Releases Page :** https://agit.ai/Polixir/revive/releases

Or pull the lastest image contains SDK and its runtime environment from Docker Hub :

    $ docker pull polixir/revive-sdk

###  Register an Account

The REVIVE SDK library is developed by Polixir. We encrypt and protect a few algorithm modules, which have their own intellectual property rights, and you can register an account to use the features of the full algorithm package.

The process  can be divided into the following two steps:

**Step 1**. Visit the REVIVE Website to register an account.

**REVIVE Website :** <https://www.revive.cn>

**Step 2**. Configure registered account information.

1.  Open the `config.yaml` file in your home directory (for example,`/home/your-user-name/.revive/.config.yaml`) in a text editor. After installing REVIVE SDK, the config.yaml file will be generated automatically.
2.  Get the accesskey from the [user center of the REVIVE Website](https://revive.cn/user-center/service) and fill it into the `config.yaml`.

``` {.yaml}
accesskey: xxxxxxxxx
```

# Usage

###  Prepare Data
Data should all be put in the `data` folder. See instruction to build your data in https://revive.cn/help/polixir-revive-sdk/index.html.

###  Train Model
`python train.py`. By default, the script will start local ray service and launch venv training and policy training both in tuning mode. You can change that behavior in command line, run `python train.py -h` for more information. For general configurations, please see `data/config.json`. For algorithm specific configurations, please see its own file. You can also modify the json file. Pass the path of the json file to the training scripts, e.g. `python train.py ... -rcf config.json` will overwrite the configurable parameters.

To run training across multiple nodes, you need to manually setup the ray cluster. Start cluster on the head node with `ray start --head --port 6379` in command line or join an existing cluster by `ray start --address=xxx --redis-password=xxx`. Pass `--address auto` when training.


# Code Structure

```
revive
├── data # data folder
│   ├── config.json
│   ├── test.npz
│   ├── test_reward.py
│   └── test.yaml
├── examples # example code              
│   ├── basic
│   ├── custom_node
│   ├── expert_function
│   ├── model_inference
│   ├── multiple_transition
│   └── parameter_tuning
├── README.md
├── revive # source code folder
│   ├── algo
│   ├── computation
│   ├── conf
│   ├── data
│   ├── dist
│   ├── __init__.py
│   ├── server.py
│   ├── utils
│   └── version.py
├── setup.py # installation script
├── tests # test scripts
│   ├── test_dists.py
│   └── test_processor.py
└── train.py # main start script
```

To add a new algorithm, you only needs to inherit the base class and implement `model_creator`, `optimizer_creator`, `train_batch` and `validate_batch`, then the code is ready to run. You can use the existing implementation as references.


#  Included Algorithms

###  Venv Training
- **Behavior Cloning**  This is a supervised learning method through neural networks.
- **Revive** derived from [Shi *.et al.* Virtual-Taobao: Virtualizing Real-World Online Retail Environment for Reinforcement Learning](https://www.aaai.org/ojs/index.php/AAAI/article/view/4419/4297).

###  Policy Training
- **PPO** derived from [Schulman *.et al.* Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347).
- **SAC** derived from [Haarnoja *et al.* Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor](https://arxiv.org/abs/1801.01290).