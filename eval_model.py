import argparse
import os
import sys
import time

import gymnasium as gym
import yaml
import numpy as np
import pandas as pd

import robot_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register('WheelBot', robot_gym.WheelBotEnv)

def parse_args():
    parser = argparse.ArgumentParser(description="Show a trained model")

    # Add hyperparameters you want to control from command line
    parser.add_argument('train_file', type=str,
                        help="Path to policy file, either YAML or zip with default values (10 deg, full height change")
    parser.add_argument('algo', type=str, choices=["PPO", "SAC"], help="Algorithm to evaluate")
    parser.add_argument('--info', type=str, choices=["none", "rew", "act", "obs", "all"], default="rew",
                        help="Which type of information to output to console")
    parser.add_argument('--length', type=int, default=512, help="Length per episode")
    parser.add_argument('--deterministic', type=bool, default=False, help="Use deterministic environment")
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = args.train_file
    base_path = os.path.dirname(file_path)
    config = None
    policy_file = None
    config = {}
    try:
        if file_path.split("/")[-1].split(".")[-1] == "yaml":
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
            policy_file = os.path.join(base_path, file_path.split("/")[-1].split(".")[0] + ".zip")

            if not os.path.exists(policy_file):
                raise FileNotFoundError("Policy file not found")

        else:
            policy_file = os.path.join(base_path, file_path.split("/")[-1].split(".")[0] + ".zip")
            config['rigid'] = False
            config['max_angle'] = 10
            config['height_level'] = 1.0
            config['duration_disturbance'] = 5
            config['first_disturbance'] = 100
            config['disturbance_window'] = 1.5
            config['max_disturbance'] = 100

            config['healthy_reward'] = 1
            config['y_angle_pen'] = 0.2
            config['y_angle_scale'] = 1.0
            config['z_angle_pen'] = 0.025
            config['z_angle_scale'] = 1.0
            config['dist_pen'] = 0.2
            config['dist_scale'] = 15.0
            config['wheel_speed_pen'] = 0.1
            config['wheel_speed_scale'] = 1.0
            config['x_vel_pen'] = 0.45
            config['x_vel_scale'] = 15.0
            config['y_angle_vel_pen'] = 0.025
            config['y_angle_vel_scale'] = 1.0

            if not os.path.exists(policy_file):
                raise FileNotFoundError("Policy file not found")

    except FileNotFoundError:
        print("The file " + str(file_path) + " does not exist, aborting!", file=sys.stderr)
        exit(1)


    env = DummyVecEnv([lambda: gym.make('WheelBot',
                            xml_file="./bot_model/wheelbot.xml",
                            render_mode="human",
                            healthy_reward=config['healthy_reward'],
                            y_angle_pen=config['y_angle_pen'],
                            y_angle_scale=config['y_angle_scale'],
                            z_angle_pen=config['z_angle_pen'],
                            z_angle_scale=config['z_angle_scale'],
                            dist_pen=config['dist_pen'],
                            dist_scale=config['dist_scale'],
                            wheel_speed_pen=config['wheel_speed_pen'],
                            wheel_speed_scale=config['wheel_speed_scale'],
                            x_vel_pen=config['x_vel_pen'],
                            x_vel_scale=config['x_vel_scale'],
                            y_angle_vel_pen=config['y_angle_vel_pen'],
                            y_angle_vel_scale=config['y_angle_vel_scale'],
                            eval = args.deterministic,
                            rigid = config['rigid'],
                            max_angle = config['max_angle'],
                            height_level = config['height_level'],
                            duration_disturbance=config['duration_disturbance'],
                            first_disturbance=config['first_disturbance'],
                            disturbance_window=config['disturbance_window'],
                            max_disturbance=config['max_disturbance'],

                            difficulty_start = 1.0,
                            frame_skip=1, width=1000, height=1000)])

    if args.algo == "PPO":
        model = PPO.load(policy_file, device="cpu")
    elif args.algo == "SAC":
        model = SAC.load(policy_file, device="cpu")
    else:
        print("The algo " + config["algo"] + "does not exist, aborting!", file=sys.stderr)
        env.close()
        exit(1)

    try:
        while True:
            obs = env.reset()
            rew = 0
            logs = []

            for _ in range(100):
                env.render()
                time.sleep(0.01)
            for step in range(args.length):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                rew += reward  # Accumulate reward over the episode

                # Handle vectorized environments or non-scalar outputs
                reward_scalar = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                info_dict = info[0] if isinstance(info, list) else info
                obs_array = obs[0] if isinstance(obs, np.ndarray) else obs
                action_array = action[0] if isinstance(action, np.ndarray) else action

                # log per-step data
                row = {"step": step, "reward": reward_scalar,}
                #row.update({f"obs_{i}": val for i, val in enumerate(obs_array)})  # no names = bad
                row.update({
                    "x": obs_array[0],              # x position
                    "y": obs_array[1],              # y position
                    "z": obs_array[2],              # z position (TODO: wirklich?)
                    "x_angle": obs_array[3],        # angle around x-axis (TODO: wirklich?)
                    "y_angle": obs_array[4],        # angle around y-axis
                    "z_angle": obs_array[5],        # angle around z-axis
                    "wheel_speed_l": obs_array[6],  # Left wheel rotational speed
                    "wheel_speed_r": obs_array[7],  # Right wheel rotational speed
                    "x_vel": obs_array[8],          # Linear velocity in x-direction
                    "y_angle_vel": obs_array[9],    # Angular velocity around y-axis
                })
                row.update({f"action_{i}": val for i, val in enumerate(action_array)})  # TODO: name correctly; "wheel
                # activation?"
                # Individual rewards (already penalized with the penalty factor i [0,1])
                row.update({k: float(v) if isinstance(v, np.generic) else v for k, v in info_dict.items()})

                logs.append(row)

                if args.info == "rew":
                    print("Reward: " + str(reward) + " | "+ str(info))
                elif args.info == "act":
                    print("Action: " + str(action))
                elif args.info == "obs":
                    print("Obs: " + str(obs))
                elif args.info == "all":
                    print("Reward: " + str(reward) + " | " + str(info))
                    print("Action: " + str(action))
                    print("Obs: " + str(obs))

                if done:
                    break
                time.sleep(0.01)

            # Save the logs of that episode (warning: overwrites old logs)
            df = pd.DataFrame(logs)
            df.to_csv("eval_logs/eval_log.csv", index=False)
            print("Saved eval_log.csv with shape:", df.shape)

            print(rew)
            time.sleep(2)

    except KeyboardInterrupt:
        print("Exiting...")
        env.close()
        exit(0)

if __name__ == '__main__':
    main()
