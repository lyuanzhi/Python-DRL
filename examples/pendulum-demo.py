from DRL.td3 import TD3
from DRL.sac import SAC
from DRL.ddpg import DDPG
import gym
import numpy as np
import argparse
import cv2

env = gym.make('Pendulum-v1', render_mode = "rgb_array")

N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]
ACTION_SHAPE = env.action_space.shape
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high

def train(type):
    if type == "DDPG":
        EPOCH = 200
        agent = DDPG(N_STATES, N_ACTIONS, MAX_ACTION)
    if type == "TD3":
        EPOCH = 300
        agent = TD3(N_STATES, N_ACTIONS, MAX_ACTION)
    if type == "SAC":
        EPOCH = 3000
        agent = SAC(N_STATES, N_ACTIONS, MAX_ACTION)
    ep_r = 0
    var = 1
    for i in range(EPOCH):
        s = env.reset()[0]
        while True:
            # explore
            noise = np.random.normal(0, var, size=ACTION_SHAPE)
            a = (agent.select_action(s) + noise).clip(ACTION_LOW, ACTION_HIGH)
            s_, r, d, trunc, _ = env.step(a)
            agent.store_data(s, a, r, s_, d)
            s = s_
            ep_r += r
            if d or trunc:
                break
        agent.learn()
        var *= 0.99
        if i % 10 == 0 and i != 0:
            print("# episode :{}, avg score : {:.1f}".format(i, ep_r / 10))
            ep_r = 0
    agent.save()

def test(type):
    if type == "DDPG":
        agent = DDPG(N_STATES, N_ACTIONS, MAX_ACTION, is_train=False, is_load=True)
    if type == "TD3":
        agent = TD3(N_STATES, N_ACTIONS, MAX_ACTION, is_train=False, is_load=True)
    if type == "SAC":
        agent = SAC(N_STATES, N_ACTIONS, MAX_ACTION, is_train=False, is_load=True)
    while True:
        s = env.reset()[0]
        ep_r = 0
        while True:
            a = agent.select_action(s)
            s_, r, d, trunc, _ = env.step(a)
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test",img)
            cv2.waitKey(1)
            s = s_
            ep_r += r
            if d or trunc:
                break
        print("# avg score : {:.1f}".format(ep_r))

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True, type=str, help='"train" or "test"')
parser.add_argument('-t', '--type', required=True, type=str, help='"DDPG" or "TD3" or "SAC"')
args = parser.parse_args()

if args.mode == "train":
    train(args.type)
else:
    test(args.type)
