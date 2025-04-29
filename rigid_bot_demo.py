import math


import gymnasium as gym  # open ai gym
import time

import robot_gym



def main():
    gym.register('WheelBot', robot_gym.WheelBotEnv)

    env = gym.make("WheelBot", xml_file="./bot_model/wheelbot_rigid.xml", reset_noise_scale=0.0, render_mode="human", frame_skip=1, width=1000, height=1000)
    #env = gym.make("WheelBot", xml_file="./bot_model/wheelbot.xml", reset_noise_scale=0.0, render_mode="human", frame_skip=1, width=1000, height=1000)
    #env.render() # call this before env.reset, if you want a window showing the environment

    env.reset()  # should return a state vector if everything worked
    env.render()

    a_rigid = [0.0, 0.0]
    while True:
        env.reset()
        for i in range(100):
            env.render()
            time.sleep(0.01)

        for i in range(100):
            obs, rew, _, _, info = env.step(a_rigid)
            print(info)
            time.sleep(0.01)


if __name__ == '__main__':
    main()