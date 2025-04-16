import gymnasium as gym  # open ai gym
import time

import robot_gym



def main():
    gym.register('WheelBot', robot_gym.WheelBotEnv)

    env = gym.make("WheelBot", reset_noise_scale=0.1, render_mode="human")
    #env.render() # call this before env.reset, if you want a window showing the environment

    env.reset()  # should return a state vector if everything worked
    env.render()
    while True:
        env.step(env.action_space.sample())
        time.sleep(0.05)


if __name__ == '__main__':
    main()