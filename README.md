# Automating Reinforcement Learning with Example-based Resets

### Accepted for publication in the IEEE Robotics and Automation Letters (RA-L)

Source code for reproducing the simulation results for the proposed algorithm of the paper ["Automating Reinforcement Learning with Example-based Resets"](https://arxiv.org/abs/2204.02041).

The instructions below were tested on Ubuntu 18.04, but should work on other Linux distros as well.

## Installation

Download the source code to the current user's home directory. The contents of this folder should be under ```~/autoreset_rl/```.

### 1. Install Conda package manager
Conda package manager is required for installing python dependencies. Follow the link below to install conda:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### 2. Create a Conda environment

Follow the link below to set up pre-requisites for mujoco-py:

https://github.com/openai/mujoco-py#install-mujoco


```bash
cd ~/autoreset_rl
conda env create --file ./conda_env.yml
```

If you have any issues related to MuJoCo or OpenAI Gym when setting up the conda environment, please refer to the following links:

https://github.com/openai/mujoco-py#troubleshooting

https://github.com/openai/gym

## Running experiments

Activate the conda environment.

```bash
cd ~/autoreset_rl
conda activate autoreset_rl
which python
```

Minimal (no logging) terminal commands to run the code:

```bash
python main.py --config_dir ./experiment_configs/cliff-cheetah.json
python main.py --config_dir ./experiment_configs/cliff-walker.json
python main.py --config_dir ./experiment_configs/peg-insertion_insert.json
python main.py --config_dir ./experiment_configs/peg-insertion_remove.json
```

Additional arguments are available (--logging, --record, --evaluation). Terminal command to view arguments:

```bash
python main.py --help
```

If the contents of this folder are not under ```~/autoreset_rl/```, please modify the experiment config files (JSON) accordingly.

## BibTeX

```bibtex
To appear
```
