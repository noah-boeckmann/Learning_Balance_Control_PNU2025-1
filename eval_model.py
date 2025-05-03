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
                            reset_noise_scale=1,
                            render_mode="human",
                            frame_skip=1, width=1000, height=1000)])
    #model = PPO.load("trained_models/10deg_rigid_policy.zip", device="cpu")
    model = PPO.load("temp_models/ppo_inverted_pendulum_5500000_steps.zip", device="cpu")

    while True:
        obs = env.reset()
        rew = 0
        for _ in range(1024):
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