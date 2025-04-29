import math


import gymnasium as gym  # open ai gym
import time

import robot_gym



def main():
    gym.register('WheelBot', robot_gym.WheelBotEnv)

    env = gym.make("WheelBot", xml_file="./bot_model/wheelbot.xml", reset_noise_scale=0.0, render_mode="human", frame_skip=1, width=1000, height=1000)
    #env = gym.make("WheelBot", xml_file="./bot_model/wheelbot.xml", reset_noise_scale=0.0, render_mode="human", frame_skip=1, width=1000, height=1000)
    #env.render() # call this before env.reset, if you want a window showing the environment

    env.reset()  # should return a state vector if everything worked
    env.render()

    while True:
        env.reset()
        for i in range(100):
            env.render()
            time.sleep(0.01)

        h_range = 0.5 * 0.872665
        for t in range(1000):
            a = h_range + h_range * math.sin(math.pi * t * 0.01 * 0.25 + math.pi * 1.5)
            action = [-a, a, a, -a, 0.0, 0.0]
            env.step(action)
            time.sleep(0.01)


if __name__ == '__main__':
    main()