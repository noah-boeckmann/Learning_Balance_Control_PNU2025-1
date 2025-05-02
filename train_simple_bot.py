import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import robot_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

import argparse


gym.register('WheelBot', robot_gym.WheelBotEnv)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on WheelBot Environment")

    # Add hyperparameters you want to control from command line
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--n_steps', type=int, default=1024, help='Number of steps to run per environment per update')
    parser.add_argument('--total_timesteps', type=int, default=5_000_000, help='Total timesteps for training')
    parser.add_argument('--device', type=str, default="cpu", help='Device: cpu or cuda')
    parser.add_argument('--policy', type=str, default="MlpPolicy", help='Policy architecture')
    parser.add_argument('--tensorboard_log', type=str, default="./ppo_logs", help='Tensorboard log directory')
    parser.add_argument('--save_path', type=str, default="./models/", help='Directory to save models')
    parser.add_argument('--checkpoint_freq', type=int, default=31250, help='How often to save model checkpoints')

    return parser.parse_args()

def make_env(rank, seed=0, render_mode=None, frame_skip=1):
    def _init():
        env = gym.make('WheelBot', max_episode_steps=3072,
                        xml_file="./bot_model/wheelbot_rigid.xml",
                        reset_noise_scale=0.0,
                        render_mode=render_mode,
                        frame_skip=frame_skip, width=1000, height=1000)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    args = parse_args()

    print("\n====================")
    print("Starting Training with Hyperparameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("====================\n")


    if args.num_envs == 1:
        env = SubprocVecEnv([make_env(i, render_mode="human") for i in range(args.num_envs)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])

    env = VecMonitor(env)

    # Optional: save model checkpoints during training
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_path,
        name_prefix='ppo_inverted_pendulum'
    )

    # Create PPO model
    model = PPO(
        args.policy,
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        device=args.device,
        n_steps=args.n_steps
    )
    # Train the model
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save final model
    model.save(args.save_path + "ppo_inverted_pendulum_final")

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