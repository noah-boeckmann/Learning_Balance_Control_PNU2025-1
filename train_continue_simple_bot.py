import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import robot_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

gym.register('WheelBot', robot_gym.WheelBotEnv)


def make_env(rank, seed=0, render_mode=None, frame_skip=1):
    def _init():
        env = gym.make('WheelBot', max_episode_steps=2048,
                        xml_file="./bot_model/wheelbot_rigid.xml",
                        reset_noise_scale=1,
                        render_mode=render_mode,
                        frame_skip=frame_skip, width=1000, height=1000)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():

    num_envs = 16


    if num_envs == 1:
        env = SubprocVecEnv([make_env(i, render_mode="human") for i in range(num_envs)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    env = VecMonitor(env)

    # Optional: save model checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=31250, save_path='temp_models/',
                                             name_prefix='ppo_inverted_pendulum')

    # Create PPO model
    model = PPO.load("ppo_inverted_pendulum_refined.zip", env=env, device="cpu", nsteps=2048)

    # Train the model
    model.learn(total_timesteps=20_000_000, callback=checkpoint_callback)

    # Save final model
    model.save("ppo_inverted_pendulum_refined.zip")


if __name__ == '__main__':
    main()