import argparse
import os
import sys
import time

import gymnasium as gym
import yaml

import robot_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register('WheelBot', robot_gym.WheelBotEnv)

def parse_args():
    parser = argparse.ArgumentParser(description="Show a trained model")

    # Add hyperparameters you want to control from command line
    parser.add_argument('train_file', type=str, help="Path to policy file")
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = args.train_file
    base_path = os.path.dirname(file_path)
    config = None
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("The file " + str(file_path) + " does not exist, aborting!", file=sys.stderr)
        exit(1)

    policy_file = os.path.join(base_path, file_path.split("/")[-1].split(".")[0])

    env = DummyVecEnv([lambda: gym.make('WheelBot',
                            xml_file="./bot_model/wheelbot.xml",
                            render_mode="human",
                            eval = False, # yes, this is weird. We want to see random starting positions here
                            rigid = config['rigid'],
                            max_angle = config['max_angle'],
                            height_level = config['height_level'],
                            difficulty_start = 1.0,
                            frame_skip=1, width=1000, height=1000)])

    if config["algo"] == "PPO":
        model = PPO.load(policy_file, device="cpu")
    elif config["algo"] == "SAC":
        model = SAC.load(policy_file, device="cpu")
    else:
        print("The algo " + config["algo"] + "does not exist, aborting!", file=sys.stderr)
        env.close()
        exit(1)

    while True:
        obs = env.reset()
        rew = 0
        for _ in range(100):
            env.render()
            time.sleep(0.01)
        for _ in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rew += reward
            print(info)
            time.sleep(0.01)
            if done:
                print(rew)
                time.sleep(5)
                break
if __name__ == '__main__':
    main()
