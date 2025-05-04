import gymnasium as gym
import robot_gym

gym.register('WheelBot', robot_gym.WheelBotEnv)

import os
import sys
import yaml
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on WheelBot Environment")

    # Add hyperparameters you want to control from command line
    parser.add_argument('train_name', type=str, help="Name of the training, a YAML file with the given name has to exist!")
    parser.add_argument('--cont_train', type=str, default=None, help="Path of the model to be worked on")
    parser.add_argument('--device', type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments")
    parser.add_argument('--tensorboard_log', type=str, default="tb_logs", help="Tensorboard log directory")
    parser.add_argument('--base_path', type=str, default="training", help="Base directory for training")

    return parser.parse_args()

def prepare_training(args : argparse.Namespace):
    print("\n====================")
    print("Starting Training with Parameters:")

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    base_path = os.path.join(os.getcwd(), args.base_path)
    file_path = os.path.join(base_path, args.train_name + '.yaml')
    config = None
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("The file " + str(file_path) + " does not exist, aborting!", file=sys.stderr)
        exit(1)

    print("\nand hyperparameters from file:")
    print(config)

    print("====================\n")
    return base_path, config

def make_env(rank, config:dict, seed=0, render_mode=None):
    def _init():
        env = gym.make('WheelBot', max_episode_steps=config["max_ep_steps"],
                        xml_file="./bot_model/wheelbot_rigid.xml",
                        render_mode=render_mode,
                        width=1000, height=1000,
                        healthy_reward = config['healthy_reward'],
                        y_angle_pen = config['y_angle_pen'],
                        z_angle_pen = config['z_angle_pen'],
                        dist_pen = config['dist_pen'],
                        wheel_speed_pen = config['wheel_speed_pen'],
                        reset_noise_scale = config['reset_noise_scale'],
                        difficulty_start = config['difficulty_start'],
                        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    args = parse_args()
    base_path, config = prepare_training(args)

    # Environment setup
    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0, config, render_mode="human")])
    else:
        env = SubprocVecEnv([make_env(i, config) for i in range(args.num_envs)])
    env = VecMonitor(env)

    config["difficulty_start"] = 1.0
    eval_env = DummyVecEnv([make_env(999, config)])
    eval_env = VecMonitor(eval_env)

    # Generate a folder for the training checkpoints
    checkpoint_path = os.path.join(base_path, args.train_name + "_checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["save_interval"] // args.num_envs, 1),
        save_path=checkpoint_path,
        name_prefix=args.train_name,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(config["eval_freq"] // args.num_envs, 1),
        n_eval_episodes=config["n_eval_ep"],
        log_path=checkpoint_path,
        best_model_save_path=checkpoint_path,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create PPO model
    # TODO: Implement training continuation
    model = PPO(
        config["policy"],
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        device=args.device,
        n_steps=config["n_steps"],
        learning_rate=config["learning_rate"],
    )

    try:
        # Train the model
        model.learn(total_timesteps=config["total_timesteps"], callback=callback, tb_log_name=args.train_name)
    except KeyboardInterrupt:
        print("Training interrupted, exiting...")

    # Save final model
    save_path = os.path.join(base_path, args.train_name + ".zip")
    print("Saving final model to: " + save_path)
    model.save(save_path)

    env.close()
    exit(0)


if __name__ == '__main__':
    main()