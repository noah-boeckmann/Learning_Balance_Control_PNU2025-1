import gymnasium as gym
import numpy as np

import robot_gym
from CurriculumCallback import CurriculumCallback

gym.register('WheelBot', robot_gym.WheelBotEnv)

import os
import sys
import yaml
import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on WheelBot Environment")

    # Add hyperparameters you want to control from command line
    parser.add_argument('train_file', type=str, help="Path to YAML config file of the training")
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

    base_path = args.base_path if os.path.isabs(args.base_path) else os.path.join(os.getcwd(), args.base_path)
    file_path = args.train_file
    config = None
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("The file " + str(file_path) + " does not exist, aborting!", file=sys.stderr)
        exit(1)

    train_name = file_path.split("/")[-1].split(".")[0]

    print("\nand hyperparameters from file:")
    print(config)

    print("====================\n")
    return train_name, base_path, config

def make_env(rank, config:dict, seed=0, render_mode=None):
    def _init():
        env = gym.make('WheelBot', max_episode_steps=config["max_ep_steps"],
                        xml_file="./bot_model/wheelbot.xml",
                        render_mode=render_mode,
                        width=1000, height=1000,
                        healthy_reward = config['healthy_reward'],
                        y_angle_pen = config['y_angle_pen'],
                        y_angle_scale = config['y_angle_scale'],
                        z_angle_pen = config['z_angle_pen'],
                        z_angle_scale = config['z_angle_scale'],
                        dist_pen = config['dist_pen'],
                        dist_scale = config['dist_scale'],
                        wheel_speed_pen = config['wheel_speed_pen'],
                        wheel_speed_scale = config['wheel_speed_scale'],
                        x_vel_pen=config['x_vel_pen'],
                        x_vel_scale = config['x_vel_scale'],
                        y_angle_vel_pen=config['y_angle_vel_pen'],
                        y_angle_vel_scale = config['y_angle_vel_scale'],
                        max_angle = config['max_angle'],
                        rigid = config['rigid'],
                        height_level = config['height_level'],
                        difficulty_start = config['difficulty_start'],
                        duration_disturbance=config['duration_disturbance'],
                        first_disturbance=config['first_disturbance'],
                        disturbance_window=config['disturbance_window'],
                        max_disturbance=config['max_disturbance'],
                        )
        env.reset(seed=seed + rank)
        return env
    return _init

def SAC_warmup(env, model, warmup_policy, num_envs, buffer_size):
    expert_data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "dones": [],
        "infos": [],
    }

    obs = env.reset()
    for i in range(int(buffer_size / num_envs)):
        actions, _ = warmup_policy.predict(obs, deterministic=True)
        next_obs, rewards, dones, infos = env.step(actions)

        for i in range(num_envs):
            expert_data["obs"].append(obs)
            expert_data["actions"].append(actions)
            expert_data["rewards"].append([rewards])
            expert_data["next_obs"].append(next_obs)
            expert_data["dones"].append([dones])
            expert_data["infos"].append(infos)

        obs = next_obs

    for k in expert_data:
        expert_data[k] = np.array(expert_data[k])

    for i in range(len(expert_data["obs"])):
        model.replay_buffer.add(
            expert_data["obs"][i],
            expert_data["next_obs"][i],
            expert_data["actions"][i],
            expert_data["rewards"][i],
            expert_data["dones"][i],
            expert_data["infos"][i]
        )

def main():
    args = parse_args()
    train_name, base_path, config = prepare_training(args)

    # Environment setup
    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0, config, render_mode="human")])
    else:
        env = SubprocVecEnv([make_env(i, config) for i in range(args.num_envs)])
    env = VecMonitor(env)

    config["eval"] = False # Enable the evaluation to behave randomly, this is different from what we initially wanted
    config["difficulty_start"] = 1.0
    eval_env = DummyVecEnv([make_env(999, config)])
    eval_env = VecMonitor(eval_env)

    # Generate a folder for the training checkpoints
    checkpoint_path = os.path.join(base_path, train_name + "_checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["save_interval"] // args.num_envs, 1),
        save_path=checkpoint_path,
        name_prefix=train_name,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(config["eval_freq"] // args.num_envs, 1),
        n_eval_episodes=config["n_eval_ep"],
        log_path=checkpoint_path,
        best_model_save_path=checkpoint_path,
    )

    # Curriculum Learning
    curriculum_callback = CurriculumCallback(env, config)

    callback = CallbackList([checkpoint_callback, eval_callback, curriculum_callback])

    # Create model
    if args.cont_train is not None:
        print("Continuing " + config["algo"] + " training with policy file " + args.cont_train)
        if config["algo"] == "PPO":
            model = PPO.load(args.cont_train,
                             env=env,
                             verbose=1,
                             tensorboard_log=args.tensorboard_log,
                             device=args.device,
                             n_steps=config["n_steps"],
                             learning_rate=config["learning_rate"],
                             )
        elif config["algo"] == "SAC":
            model = SAC.load(args.cont_train,
                             env=env,
                             verbose=1,
                             tensorboard_log=args.tensorboard_log,
                             device=args.device,
                             learning_rate=config["learning_rate"],
                             buffer_size=config["buffer_size"],
                             batch_size=config["batch_size"],
                             learning_starts=config["learning_starts"],
                             )
        else:
            print("The algo " + config["algo"] + "does not exist, aborting!", file=sys.stderr)
            env.close()
            eval_env.close()
            exit(1)
    else:
        print("Starting a fresh " + config["algo"] + " training")
        if config["algo"] == "PPO":
            model = PPO(
                config["policy"],
                env,
                verbose=1,
                tensorboard_log=args.tensorboard_log,
                device=args.device,
                n_steps=config["n_steps"],
                learning_rate=config["learning_rate"],
            )
        elif config["algo"] == "SAC":
            model = SAC(config["policy"],
                        env,
                        verbose=1,
                        tensorboard_log=args.tensorboard_log,
                        device=args.device,
                        learning_rate=config["learning_rate"],
                        buffer_size=config["buffer_size"],
                        batch_size=config["batch_size"],
                        learning_starts=config["learning_starts"],
                        )

            if config["warmup"] != "":
                print("Warming up with policy: " + config["warmup"])
                warmup_policy = PPO.load(config["warmup"],
                                 env=env,
                                 verbose=1,
                                 tensorboard_log=args.tensorboard_log,
                                 device="cpu",
                                 )
                SAC_warmup(env, model, warmup_policy, args.num_envs, config['buffer_size'])

    try:
        # Train the model
        model.learn(total_timesteps=config["total_timesteps"], callback=callback, tb_log_name=train_name)
    except KeyboardInterrupt:
        print("Training interrupted, exiting...")

    # Save final model
    save_path = os.path.join(base_path, train_name + ".zip")
    print("Saving final model to: " + save_path)
    model.save(save_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config["n_eval_ep"],
        render=False,
        deterministic=True,
        return_episode_rewards=False,
    )
    print(f"Final evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    exit(0)


if __name__ == '__main__':
    main()
