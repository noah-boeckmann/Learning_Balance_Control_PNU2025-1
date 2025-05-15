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
                            reset_noise_scale=1.0,
                            difficulty_start=1.0,
                            render_mode="human",
                            frame_skip=1, width=1000, height=1000)])
    #model = PPO.load("trained_models/10deg_rigid_policy.zip", device="cpu")
    model = PPO.load("training/test_sig_earlier_diff_start_cont_checkpoints/test_sig_earlier_diff_start_cont_2000000_steps.zip", device="cpu")

    while True:
        obs = env.reset()
        rew = 0
        for _ in range(1000):
            env.render()
            time.sleep(0.01)
        for _ in range(512):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rew += reward
            print(info)
            time.sleep(0.1)
            if done:
                print(rew)
                time.sleep(5)
                break
if __name__ == '__main__':
    main()