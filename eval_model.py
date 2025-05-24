import time

import gymnasium as gym
import robot_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    gym.register('WheelBot', robot_gym.WheelBotEnv)
    env = DummyVecEnv([lambda: gym.make('WheelBot',
                            xml_file="./bot_model/wheelbot_rigid.xml",
                            reset_noise_scale=1.0,
                            difficulty_start=1.0,
                            render_mode="human",
                            frame_skip=1, width=1000, height=1000)])
    print("Loading model...")
    # model = PPO.load("trained_models/10deg_rigid_policy.zip", device="cpu")
    # model = PPO.load("training/basic_checkpoints/basic_10000000_steps.zip", device="cpu")
    # model = PPO.load("training/LRate_HRew_checkpoints/best_model.zip", device="cpu")
    # model = PPO.load("training/LRate_DPen_WSpeed_checkpoints/best_model.zip", device="cpu")
    # model = SAC.load("training/base_checkpoints/base_3500000_steps.zip", device="cpu")
    model = SAC.load("training/base_SAC_checkpoints/best_model.zip", device="cpu")
    print("Model loaded. Starting evaluation...")

    while True:
        obs = env.reset()
        rew = 0
        for _ in range(100):
            env.render()
            time.sleep(0.01)
        for _ in range(5096):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rew += reward
            print(info)
            time.sleep(0.05)
            if done:
                print("Reward: ", rew)
                time.sleep(0.1)
                break
if __name__ == '__main__':
    main()