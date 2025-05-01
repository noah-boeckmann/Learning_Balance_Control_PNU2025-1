# Learning Balance Control PNU2025-1

This is the project repository for our course "Artificial Intelligence in Robotics"
at the Pusan National University, South Korea.

## Goal

The idea for the project came from the following paper:
[Balance Control of a Novel Wheel-legged Robot: Design and Experiments](https://ieeexplore.ieee.org/document/9561579)

We want to replace the LQR controller for keeping the robot upright with a learned policy.

Our project goals are as follows:

1. Setup of the simulation environment and integration with the machine learning framework. ✅
2. Definition of the robot geometry and training a proof of concept controller for a simple and static
robot state. ✅
3. Training a controller to balance the robot in its upright equilibrium position while allowing for
changing height. 🔜
4. Improving the controller further by introducing perturbations such as shifting the center of gravity. 🔜

## Model

We approximated the robot in the MuJoCo simulation environment by estimating the dimensions from one
of the papers pictures and Table I:

![bot geometry](bot_model/bot_geometry.png)
![bot model](bot_model/bot_model.png)

## Results

We were able to train a basic policy for the robot with no height change and no other perturbations:
[Basic Rigid Policy](trained_model/basic_rigid_policy.zip)
![basic rigid policy](trained_models/basic_rigid_policy.png)
Reward function:
$`reward = \text{alive} - 0.1 * \text{y\_angle}^2 - 0.1 * \text{x\_angle}^2 - 0.5 * (\text{wheel\_speed\_l}^2 + \text{wheel\_speed\_r}^2) - 10 * \text{x\_dist}`$
