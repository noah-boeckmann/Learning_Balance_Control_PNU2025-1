from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


class WheelBotEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description --- Taken from the gym mujoco reference implementation, subject to change!
    This environment takes the wheelbot model and makes it trainable.


    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical torque applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|------------|
    | 0   | Torque applied on the cart| -1          | 1           | wheel_r / wheel_l                | Torque Nm  |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (1 element):* Position values of the robot's centerpoint
    - *qvel (1 element):* Velocity values of the robot's centerpoint'
    - *angle of the robot with regard to the world frame
    - *angle velocity of the robot with regard to the world frame
    - *leg_angle current angle of the legs

    The observation space is a `Box(-Inf, Inf, (9,), float64)` where the elements are as follows:
    TODO:
    | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   |                                                                   | -Inf | Inf |                                  |       | position (m)             |
    | 1   |                                                                   | -Inf | Inf |                                  |       | unitless                 |
    | 2   |                                                                   | -Inf | Inf |                                  |       | unitless                 |
    | 3   |                                                                   | -Inf | Inf |                                  |       | unitless                 |
    | 4   |                                                                   | -Inf | Inf |                                  |       | unitless                 |
    | 5   |                                                                   | -Inf | Inf |                                  |       | velocity (m/s)           |



    ## Rewards
    The total reward is: ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*.

    - *alive_bonus*:
    Every timestep that the robot is healthy, it gets a reward of fixed value `healthy_reward` (default is $10$).
    - *distance_penalty*:
    This reward is a measure of how far the robot has moved from the (0, 0, -) point in the world frame

    - *velocity_penalty*:
    A negative reward to penalize the agent for moving too fast.

    `info` contains the individual reward terms.


    ## Starting State

    ## Episode End
    ### Termination
    The environment terminates when the Robot is unhealthy.
    The Robot is unhealthy if any of the following happens:

    1.Termination: The absolute angle of the robots center point with regard to the world frame is greater than 45 degrees.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "./bot_model/wheelbot.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)

        self._healthy_reward = healthy_reward
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        y_angle = observation[4]
        terminated = bool(abs(y_angle) > 30)
        reward, reward_info = self._get_rew(observation, terminated)

        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, observation, terminated):
        x, y = observation[0], observation[1]
        y_angle = observation[4]
        #dist_penalty = 0.01 * x**2 + y ** 2
        dist_penalty = 0.0
        angle_penalty = np.exp(0.1 * abs(y_angle)) + 1
        alive_bonus = self._healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - angle_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "angle_penalty": -angle_penalty,
        }

        return reward, reward_info

    def _get_obs(self):
        quat = self.data.xquat[1]  # [w, x, y, z]
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # convert to [x, y, z, w]
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)

        return np.concatenate([self.data.xpos[1],  # bot pos + angle
                euler])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            ),
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale,
        )
        return self._get_obs()
