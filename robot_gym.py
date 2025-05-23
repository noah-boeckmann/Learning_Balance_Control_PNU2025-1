from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

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
    | 8   | Velocity of box in X                                              | -Inf | Inf | vel (m/s)                 |
    | 9   | Angle velocity of box around y-axis                               | -Inf | Inf | angle vel (rad/s)    |

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
        healthy_reward: float = 20.0,
        y_angle_pen: float = 0.1,
        z_angle_pen: float = 0.1,
        dist_pen: float = 100.0,
        wheel_speed_pen: float = 0.5,
        x_vel_pen: float = 1.0,
        y_angle_vel_pen: float = 0.5,
        max_angle: float = 0.0,
        rigid: bool = False,
        height_level: float = 1.0,
        difficulty_start: float = 0.0,
        eval: bool = False,


        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, max_angle, **kwargs)

        self._healthy_reward = healthy_reward
        self._y_angle_pen = y_angle_pen
        self._z_angle_pen = z_angle_pen
        self._dist_pen = dist_pen
        self._wheel_speed_pen = wheel_speed_pen
        self._x_vel_pen = x_vel_pen
        self._y_angle_vel_pen = y_angle_vel_pen

        self._max_angle = max_angle
        self._difficulty = difficulty_start
        self._difficulty_start = difficulty_start

        self._bot_height = None # Will be calculated in reset
        self._height_actor_max = 0.872665
        self._rigid = rigid
        self.set_height_level(height_level)

        self._eval = eval


        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.action_space = Box(low=-100, high=100, shape=(2,), dtype=np.float32)

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

        action = np.concatenate([self._height_actor_action, action], axis=0)
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

    def set_height_level(self, height_level):
        self._height_level = height_level
        actor_angle = self._height_actor_max * (1 - self._height_level)
        self._height_actor_action = [-actor_angle, actor_angle, actor_angle, -actor_angle]

    def set_max_angle(self, angle):
        self._max_angle = angle

    def _get_rew(self, observation, terminated):
        x, y = observation[0], observation[1]
        y_angle = observation[4]
        z_angle = observation[5]

        # x speed and y angle speed sensors:
        x_vel = observation[8]  # Velocity in x-axis direction
        y_angle_vel = observation[9]  # Angular velocity around the y-axis

        dist_penalty = self._dist_pen * x**2 #+ 0.1 * y ** 2
        y_angle_penalty = self._y_angle_pen * (y_angle ** 2)

        # X velocity and y angle velocity penalties:
        x_vel_penalty = self._x_vel_pen * x_vel ** 2  # avoid too much movement in y
        y_angle_vel_penalty = self._y_angle_vel_pen * y_angle_vel ** 2  # avoid gier on x achsis

        wheel_speed_l = observation[6]
        wheel_speed_r = observation[7]

        wheel_l_penalty = self._wheel_speed_pen * wheel_speed_l ** 2
        wheel_r_penalty = self._wheel_speed_pen * wheel_speed_r ** 2

        # FIXME: z is not measured in the worldframe - no issue when upright but should be looked into
        z_angle_penalty = self._z_angle_pen * (z_angle ** 2)

        alive_bonus = self._healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - y_angle_penalty - wheel_l_penalty - wheel_r_penalty - z_angle_penalty - y_angle_vel_penalty - x_vel_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "y_angle_penalty": -y_angle_penalty,
            "z_angle_penalty": -z_angle_penalty,
            "wheel_l_penalty": -wheel_l_penalty,
            "wheel_r_penalty": -wheel_r_penalty,
            "y_angle_vel_penalty": -y_angle_vel_penalty,
            "x_vel_penalty": -x_vel_penalty,
        }

        return reward, reward_info

    def _get_obs(self):
        quat = self.data.xquat[1]  # [w, x, y, z]
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # convert to [x, y, z, w]
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)

        wheel_sp_l = self.data.sensordata[0]
        wheel_sp_r = self.data.sensordata[1]
        y_angle_vel = self.data.sensordata[3]
        x_vel = self.data.sensordata[5]
        sensordata = [wheel_sp_l, wheel_sp_r, x_vel, y_angle_vel]

        return np.concatenate([self.data.xpos[1],  # bot pos + angle + distance traveled by wheels
                euler, sensordata])

    def reset_model(self):
        # TODO: introduce z angle noise?

        if self._eval:
            angle = self._max_angle

        else:
            # generate a random starting angle
            angle_low = -self._max_angle * self._difficulty
            angle_high = self._max_angle * self._difficulty
            angle = self.np_random.uniform(angle_low, angle_high)

            # set a random height level if not rigid
            if not self._rigid: self.set_height_level(self.np_random.uniform(1 - self._difficulty, 1.0))

        self._bot_height, beta = self.calculate_reset_hinge_angles()
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

        # set the angle actor initial angles
        state[7], state[10], state[12], state[15] = self._height_actor_action
        self.data.ctrl = np.concatenate([self._height_actor_action, [0.0, 0.0]])

        # set the second leg hinges to their appropriate angles to avoid excessive movement at simulation start
        state[8] = beta
        state[11] = -beta
        state[13] = -beta
        state[16] = beta

        self.set_state(
            state,
            self.init_qvel,
        )
        return self._get_obs()

    def calculate_reset_hinge_angles(self):
        actor_angle = self._height_actor_action[1] # Take the positive height actor angle

        alpha = actor_angle + 0.698132 # add angle offset of joint
        a = np.cos(alpha) * 0.28 # Length of the first leg part triangle
        b = np.sqrt(0.165649 - (np.sin(alpha) * 0.28 + 0.07) ** 2) # Length of the second leg part triangle
        height = a + b # Bot height from box frame to wheel hub
        beta = alpha + np.arccos(b / 0.407) # Angle from leg part 1 to leg part 2

        height = height + 0.1 # add wheel height

        return height, beta
