import time

import gymnasium as gym
import robot_gym_height
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    gym.register('WheelBot', robot_gym_height.WheelBotEnv)
    env = DummyVecEnv([lambda: gym.make('WheelBot',
                            xml_file="./bot_model/wheelbot.xml",
                            reset_noise_scale=0.5,
                            difficulty_start=1.0,
                            render_mode="human",
                            frame_skip=1, width=1000, height=1000)])
    #model = PPO.load("trained_models/10deg_rigid_policy.zip", device="cpu")
    model = PPO.load("./best_model.zip", device="cpu")

    while True:
        obs = env.reset()
        rew = 0
        for _ in range(500):
            env.render()
            time.sleep(0.01)
        for _ in range(2048):
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
