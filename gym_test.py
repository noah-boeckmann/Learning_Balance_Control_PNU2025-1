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

    t = 0.0

    #a_act = [0.872665, -0.872665,-0.872665, 0.872665, 0.01, 0.01]
    a_rigid = [0.0, 0.0]
    while True:
        env.reset()
        for i in range(100):
            env.render()
            time.sleep(0.01)

        for i in range(100):
            obs, rew, _, _, info = env.step(a_rigid)
            print(info)
            time.sleep(0.1)

    h_range = 0.5 * (1.5708 - 0.6980)
    while True:
        a = h_range + h_range * math.sin(math.pi * t * 0.25)
        action = [-1.5708 + a, 1.5708 - a, 1.5708 - a, -1.5708 + a]
        t = t + 0.01
        env.step(action)
        time.sleep(0.01)


if __name__ == '__main__':
    main()