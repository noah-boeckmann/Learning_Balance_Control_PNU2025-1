import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import robot_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

gym.register('WheelBot', robot_gym.WheelBotEnv)


def make_env(rank, seed=0, render_mode=None):
    def _init():
        env = gym.make('WheelBot',
                        xml_file="./bot_model/wheelbot_rigid.xml",
                        reset_noise_scale=0.0,
                        render_mode=render_mode,
                        frame_skip=5, width=1000, height=1000)
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
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/',
                                             name_prefix='ppo_inverted_pendulum')

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs", device="cpu", n_steps=2048)

    # Train the model
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)

    # Save final model
    model.save("ppo_inverted_pendulum_final")

    # Evaluate
    while True:
        obs = env.reset()
        for _ in range(2048):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()