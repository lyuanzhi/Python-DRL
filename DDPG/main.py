from ddpg import DDPG
import gym
import numpy as np

env = gym.make('Pendulum-v0')

EPOCH = 200
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]
ACTION_SHAPE = env.action_space.shape
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high

def train():
    agent = DDPG(N_STATES, N_ACTIONS, MAX_ACTION)
    ep_r = 0
    var = 1
    for i in range(EPOCH):
        s = env.reset()
        while True:
            noise = np.random.normal(0, var, size=ACTION_SHAPE)
            a = (agent.select_action(s) + noise).clip(ACTION_LOW, ACTION_HIGH)
            s_, r, d, _ = env.step(a)
            agent.store_data(s, a, r, s_, d)
            # env.render()
            s = s_
            ep_r += r
            if d:
                break
        agent.learn()
        var *= 0.99
        if i % 10 == 0 and i != 0:
            print("# episode :{}, avg score : {:.1f}".format(i, ep_r / 10))
            ep_r = 0
    agent.save()

def test():
    agent = DDPG(N_STATES, N_ACTIONS, MAX_ACTION, is_train=False, is_load=True)
    while True:
        s = env.reset()
        ep_r = 0
        while True:
            a = agent.select_action(s)
            s_, r, d, _ = env.step(a)
            env.render()
            s = s_
            ep_r += r
            if d:
                break
        print("# avg score : {:.1f}".format(ep_r))

if __name__ == '__main__':
    # train()
    test()
