# Learning to use different tools for objects rearrangement
[Project Website](https://yunchuzhang.github.io/toolselector.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://yunchuzhang.github.io/images/Tool_select_IROS.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;Under review for IROS 2022

## Abstract

Object rearrangement and cleaning tasks in complex scenes require the ability to utilize different tools. It is important to correctly switch between and deploy suitable tools. Previous works focus on either mastering manipulation tasks with a single tool, or learning task-oriented grasping for single tools.
In this work, we propose an end-to-end learning framework that jointly learns to choose different tools and deploy tool-conditioned policies with a limited amount of human demonstrations. We evaluate our method on parallel gripper and suction cup picking and placing, brush sweeping, and household rearrangement tasks, generalizing to different configurations, novel objects, and cluttered scenes in the real world.

## Method Overview
<img src="https://github.com/YunchuZhang/Learning-to-use-different-tools-for-objects-rearrangement/blob/main/docs/tool_affordance.png"><br>
In order to solve a complex household rearrangement task as shown in the figure above, the policy consists of two parts: the affordance-aware tool selection policy (picking prediction module) and the selection-conditioned continuous action policy (placing prediction module). The affordance-aware tool selection module is in charge of figuring out which tool to deploy at each step and where to deploy it. In other words, it needs to be able to learn the affordance in the input image. For example, the robot could learn to first move the objects to clear the workspace for sweeping, instead of trying to sweep beans while the objects are still in the way. Given the predicted starting location, the 
second module chooses how the tool should act. We implement these policies as neural networks and train with gradient-based training algorithms.

## Installation

**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}"\n' >> ~/.profile  # Add Conda to PATH.
source ~/.profile
conda init
```

**Step 2.** Create and activate Conda environment, then install GCC and Python packages.

```shell
cd ~/ravens
conda create --name ravens python=3.7 -y
conda activate ravens
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
python setup.py install --user
```

**Step 3.** Recommended: install GPU acceleration with NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 for Tensorflow.
```shell
./oss_scripts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```

## Getting Started

**Step 1.1** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode. (For simulation only)

```shell
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=test --n=100
```

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 1.2** Processing real data to pkl format.
```shell
python ravens/get_label.py
```

**Step 2.** Train a model e.g., Tool-Transporter Networks model. Model checkpoints are saved to the `checkpoints` directory. Optional: you may exit training prematurely after 1000 iterations to skip to the next step.

```shell
python ravens/train.py --task=mix --agent=tooltransporter --n_demos=10 
```

**Step 3.** Evaluate a Tool-Transporter Networks agent using the model trained for 33000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens/test.py  --assets_root=./ravens/environments/assets/ --root_dir=. --task=sweep --agent=tooltransporter --n_demos=430 --n_steps=33000
python ravens/test.py  --assets_root=./ravens/environments/assets/ --root_dir=. --task=suc --agent=tooltransporter --n_demos=430 --n_steps=33000
python ravens/test.py  --assets_root=./ravens/environments/assets/ --root_dir=. --task=mix --agent=tooltransporter --n_demos=430 --n_steps=33000
```

**Step 4.** Plot and print results.

```shell
python ravens/plot.py --disp=True --task=block-insertion --agent=transporter --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

## Datasets and Pre-Trained Models

Download our generated train and test datasets and pre-trained models [here](https://drive.google.com/drive/folders/1WZR6Npqiy-1DZNaWiN3yOI1hQnpovAww?usp=sharing).


The MDP formulation for each task uses transitions with the following structure:

**Observations:** raw RGB-D images and camera parameters (pose and intrinsics).

**Actions:** a primitive function (to be called by the robot) and parameters.

**Rewards:** total sum of rewards for a successful episode should be =1.

**Info:** 6D poses, sizes, and colors of objects.
