# Learning Balance Control PNU2025-1

This is the project repository for our course "Artificial Intelligence in Robotics"
at the Pusan National University, South Korea.

## Content
1. Introduction
2. Physics Simulation
3. Training
4. Approach
5. Achievements



## Introduction

The idea for the project came from the following paper:
[Balance Control of a Novel Wheel-legged Robot: Design and Experiments](https://ieeexplore.ieee.org/document/9561579)

We want to replace the LQR controller for keeping the robot upright with a learned policy.

Our project goals are as follows:

1. Setup of the simulation environment and integration with the machine learning framework. ✅
2. Definition of the robot geometry and training a proof of concept controller for a simple and static
robot state. ✅
3. Training a controller to balance the robot in its upright equilibrium position while allowing for changing height. ✅
4. Improving the controller further by introducing random starting angles and applying a force impulse during the episode. ✅

## Physics Simulation


We approximated the robot in the MuJoCo simulation environment by estimating the dimensions from one
of the papers pictures and Table I:

![bot geometry](bot_model/bot_geometry.png)
![bot model](bot_model/bot_model.png)

#TODO

## Training
The training is based on the [MuJoCo Environment](https://gymnasium.farama.org/environments/mujoco/) implemented by the gymnasium project. Our robot environment is adapted from the
[MuJoCo Cartpole](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) environment.

### Environment
The environment takes arguments for configuring the behavior such as changing the robots'
height, maximum perturbation angles and forces, and other settings.

The environment implements the possibility to introduce the following perturbations
during training which are scaled by the current difficulty level to enable curriculum learning:
- Random height 
- Random y-angle upon reset
- Application of a force within a configurable time step window

The action space is a 2-dimensional action in [-10, 10] where each value represents
the torque applied by the wheels' actuator.

The observation space is the bots position and rotation, the rotational speed of the wheels,
the robots x-velocity and y-angular-velocity.

### High level training
We have implemented training for two algorithms: Stable Baselines3 [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

The configuration of a training is stored in a YAML file which has to be provided as an
argument when starting the training. Basic configuration files for [PPO](./training/basic_PPO.yaml) and [SAC](./training/basic_SAC.yaml) with all necessary
entries are provided. In addition to that there is command line arguments to further
configure the training process.

The training can be parallelized. Each additional training environment (can be set
with ``--num_envs n``) consumes around 250 MB of RAM.

A training can be started with a pretrained model to continue with different settings
or perturbations (provide path to zip with ``--cont_train X``)

The general training process is as follows:

1. Setup of training parameters
2. Initialization of the training and evaluation environment
3. Setup of the checkpoint, evaluation and curriculum callbacks
4. Preparation of the policy to train (SAC: optional warmup of the replay buffers with a pretrained PPO policy)
5. Training of the policy with checkpoints, curriculum calculation and evaluation
6. Final evaluation and policy saving (same name as YAML file)

#### Curriculum Learning
The curriculum learning callback calculates a difficulty scalar [0, 1] depending on
the current training step progress $`\frac{n_{step}}{N_{step}}`$ and sets the
difficulty in the environment(s). The difficulty function can be configured, the
default is a sigmoid function which showed the best results during our trainings.

## Approach
#TODO
## Achievements
#TODO


We were able to train a basic policy for the robot with no height change and no other perturbations:
[Basic Rigid Policy](trained_model/basic_rigid_policy.zip)
![basic rigid policy](trained_models/basic_rigid_policy.png)
Reward function:
$`\text{reward} = \text{alive} - 0.1 * \text{y\_angle}^2 - 0.1 * \text{x\_angle}^2 - 0.5 * (\text{wheel\_speed\_l}^2 + \text{wheel\_speed\_r}^2) - 10 * \text{x\_dist}`$
