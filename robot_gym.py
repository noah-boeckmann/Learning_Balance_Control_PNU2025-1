from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

import yaml

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


def _sig_of_a_quad(penalty, observation):
    return 2 / (1 + np.exp((penalty * observation**2))) - 1

class WheelBotEnv(MujocoEnv, utils.EzPickle):
    r"""
    This environment takes the wheelbot model and makes it trainable.


    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1000, 1000]`, where `action` represents the
    numerical torque applied to the robots wheels (with magnitude representing the amount of torque and
    sign representing the direction)

    | Num | Action                            | Control Min | Control Max | Name       | Type (Unit)|
    |-----|-----------------------------------|-------------|-------------|------------|------------|
    | 0   | Torque applied on the left wheel  | -1000       | 1000        | wheel_l_m  | Torque (Nm)|
    | 1   | Torque applied on the right wheel | -1000       | 1000        | wheel_r_m  | Torque (Nm)|


    ## Observation Space
    The observation space consists of the following parts (in order):

    - Position values of the robot's centerpoint [3 Elements]
    - Euler angle of the robot with regard to the world frame [3 Elements]
    - Wheel turning velocities L & R [2 Elements]

    The observation space is a `Box(-Inf, Inf, (8,), float64)` where the elements are as follows:
    | Num | Observation                                                       | Min  | Max | Type (Unit)               |
    | --- | ----------------------------------------------------------------- | ---- | --- | ------------------------- |
    | 0   | Bot X-Position                                                    | -Inf | Inf | position  (m)             |
    | 1   | Bot Y-Position                                                    | -Inf | Inf | position  (m)             |
    | 2   | Bot Z-Position                                                    | -Inf | Inf | position  (m)             |
    | 3   | Bot X-Rotation                                                    | -Inf | Inf | angle     (deg)           |
    | 4   | Bot Y-Rotation                                                    | -Inf | Inf | angle     (deg)           |
    | 5   | Bot Z-Rotation                                                    | -Inf | Inf | angle     (deg)           |
    | 6   | Wheel L rotational speed                                          | -Inf | Inf | angle vel (deg)           |
    | 7   | Wheel R rotational speed                                          | -Inf | Inf | angle vel (deg)           |

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
        bot_height: float = 0.635,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 20.0,
        y_angle_pen: float = 0.1,
        z_angle_pen: float = 0.1,
        dist_pen: float = 100.0,
        wheel_speed_pen: float = 0.5,
        reset_noise_scale: float = 0.0,
        difficulty_start: float = 0.0,

        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)

        self._healthy_reward = healthy_reward
        self._y_angle_pen = y_angle_pen
        self._z_angle_pen = z_angle_pen
        self._dist_pen = dist_pen
        self._wheel_speed_pen = wheel_speed_pen

        self._reset_noise_scale = reset_noise_scale
        self._difficulty = difficulty_start
        self._difficulty_start = difficulty_start
        self._bot_height = bot_height

        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)

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

    def set_difficulty(self, difficulty):
        self._difficulty = max(self._difficulty_start, difficulty)

    def _get_rew(self, observation, terminated):

        # config = None
        # try:
        #     with open(file_path, 'r') as file:
        #         config = yaml.safe_load(file)
        # except FileNotFoundError:
        #     print("The file " + str(file_path) + " does not exist, aborting!", file=sys.stderr)
        #     exit(1)

        # OLD REWARD FUNCTION
        # x, y = observation[0], observation[1]
        # z_angle = observation[5]
        # y_angle = observation[4]
        # wheel_speed_l = observation[6]
        # wheel_speed_r = observation[7]
        #
        # dist_penalty = self._dist_pen * x ** 2  # + 0.1 * y ** 2
        #
        # # y_angle_penalty = min(100, 3.5 * np.exp(0.2 * abs(y_angle)) - 3.5)
        # # y_angle_penalty = min(100, 0.4 * (y_angle ** 2))
        # y_angle_penalty = self._y_angle_pen * (y_angle ** 2)
        #
        # wheel_l_penalty = self._wheel_speed_pen * wheel_speed_l ** 2
        # wheel_r_penalty = self._wheel_speed_pen * wheel_speed_r ** 2
        #
        # # FIXME: z is not measured in the world (reference) frame - no issue when upright but should be looked into
        # z_angle_penalty = self._z_angle_pen * (z_angle ** 2)
        #
        # alive_bonus = self._healthy_reward * int(not terminated)
        #
        # reward = (
        #         alive_bonus
        #         - dist_penalty
        #         - y_angle_penalty
        #         - wheel_l_penalty
        #         - wheel_r_penalty
        #         - z_angle_penalty
        # )

        # Attempt
        # x, y = observation[0], observation[1]
        # z_angle = observation[5]
        # y_angle = observation[4]
        # wheel_speed_l = observation[6]
        # wheel_speed_r = observation[7]
        #
        # dist_penalty = self._dist_pen * np.tanh(x ** 2 + y ** 2)
        # y_angle_penalty = self._y_angle_pen * np.tanh(y_angle ** 2)
        # # wheel_l_penalty = self._wheel_speed_pen * np.tanh(abs(wheel_speed_l))
        # # wheel_r_penalty = self._wheel_speed_pen * np.tanh(abs(wheel_speed_r))
        # wheel_l_penalty = self._wheel_speed_pen * np.tanh(wheel_speed_l ** 2)
        # wheel_r_penalty = self._wheel_speed_pen * np.tanh(wheel_speed_r ** 2)
        # z_angle_penalty = self._z_angle_pen * np.tanh(z_angle ** 2)  # FIXME: z is not measured in the world (
        # # reference) frame - no issue when upright but should be looked into
        #
        # alive_bonus = self._healthy_reward  # Smooth alive reward (avoid discontinuities for SAC)
        #
        # # Optional: bonus for staying upright
        # # upright_bonus = 1.0 - abs(y_angle) / np.pi  # normalized
        #
        # reward = (
        #         alive_bonus
        #         # + 0.5 * upright_bonus
        #         - dist_penalty
        #         - y_angle_penalty
        #         - wheel_l_penalty
        #         - wheel_r_penalty
        #         - z_angle_penalty
        # )

        x, y = observation[0], observation[1]
        z_angle = observation[5]
        y_angle = observation[4]
        wheel_speed_l = observation[6]
        wheel_speed_r = observation[7]

        dist_penalty = _sig_of_a_quad(self._dist_pen,x)
        y_angle_penalty = _sig_of_a_quad(self._y_angle_pen,y_angle)
        wheel_l_penalty = _sig_of_a_quad(self._wheel_speed_pen,wheel_speed_l)
        wheel_r_penalty = _sig_of_a_quad(self._wheel_speed_pen,wheel_speed_r)
        z_angle_penalty = _sig_of_a_quad(self._z_angle_pen,z_angle)  # FIXME: z is not measured in the world (reference) frame - no issue when upright but should be looked into

        alive_bonus = self._healthy_reward  # Smooth alive reward (avoid discontinuities for SAC)

        # Optional: bonus for staying upright
        # upright_bonus = 1.0 - abs(y_angle) / np.pi  # normalized

        reward = (
                alive_bonus
                # + 0.5 * upright_bonus
                - dist_penalty
                - y_angle_penalty
                - wheel_l_penalty
                - wheel_r_penalty
                - z_angle_penalty
        )

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "y_angle_penalty": -y_angle_penalty,
            "z_angle_penalty": -z_angle_penalty,
            "wheel_l_penalty": -wheel_l_penalty,
            "wheel_r_penalty": -wheel_r_penalty,
        }

        return reward, reward_info

    def _get_obs(self):
        quat = self.data.xquat[1]  # [w, x, y, z]
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # convert to [x, y, z, w]
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)

        return np.concatenate([self.data.xpos[1],  # bot pos + angle + distance traveled by wheels
                euler, self.data.sensordata])

    def reset_model(self):
        # TODO: introduce an eval mode in which the angle can be set manually without RNG. Also useful for potential further perturbations
        # TODO: introduce z angle noise?
        # TODO: enable varying the height of the robot?

        noise_low = -self._reset_noise_scale * self._difficulty
        noise_high = self._reset_noise_scale * self._difficulty

        angle = self.np_random.uniform(10 * noise_low, 10 * noise_high)
        angle = angle * np.pi / 180
        quat = R.from_euler(seq="y", angles=angle).as_quat()

        # we have to compensate for the height difference so we always touch the ground at start
        # take the distance to the middle of the wheel and add the wheels radius afterwards to account for the angle error
        z_height = np.cos(angle) * (self._bot_height - 0.1) + 0.1

        state = self.init_qpos.copy()
        state[2] = z_height

        # apparently mujoco denotes quaternions [w, x, y, z] instead of [x, y, z, w]
        state[3] = quat[3]
        state[4:7] = quat[:3]

        self.set_state(
            state,
            self.init_qvel,
        )
        return self._get_obs()
