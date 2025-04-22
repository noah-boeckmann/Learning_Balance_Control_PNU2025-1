import time

import gymnasium as gym
import robot_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    gym.register('WheelBot', robot_gym.WheelBotEnv)
    env = DummyVecEnv([lambda: gym.make('WheelBot',
                            xml_file="./bot_model/wheelbot_rigid.xml",
                            reset_noise_scale=0.0,
                            render_mode="human",
                            frame_skip=1, width=1000, height=1000)])
    model = PPO.load("ppo_inverted_pendulum_final.zip")

    while True:
        obs = env.reset()
        for _ in range(2048):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            time.sleep(0.01)
            if done:
                obs = env.reset()

if __name__ == '__main__':
    main()