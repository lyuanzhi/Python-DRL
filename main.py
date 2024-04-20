from DRL.ppo import PPO
from DRL.pg import PG
from DRL.dqn import DQN
import gym

env = gym.make('CartPole-v1')

N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

def train(type):
    if type == "DQN":
        EPOCH = 200
        agent = DQN(N_STATES, N_ACTIONS)
    if type == "PPO":
        EPOCH = 200
        agent = PPO(N_STATES, N_ACTIONS)
    if type == "PG":
        EPOCH = 800
        agent = PG(N_STATES, N_ACTIONS)
    ep_r = 0
    for i in range(EPOCH):
        s = env.reset()
        while True:
            a = agent.select_action(s)
            s_, r, d, _ = env.step(a)
            if type == "DQN":
                r = (env.theta_threshold_radians - abs(s_[2])) / env.theta_threshold_radians * 0.7 + (env.x_threshold - abs(s_[0])) / env.x_threshold * 0.3
            agent.store_data(s, a, r, s_, d)
            # env.render()
            s = s_
            ep_r += r
            if type == "DQN":
                agent.learn()
            if d:
                break
        if type == "PPO" or type == "PG":
            agent.learn()
        if i % 10 == 0 and i != 0:
            print("# episode :{}, avg score : {:.1f}".format(i, ep_r / 10))
            ep_r = 0
    agent.save()

def test(type):
    if type == "DQN":
        agent = DQN(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PPO":
        agent = PPO(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PG":
        agent = PG(N_STATES, N_ACTIONS, is_train=False, is_load=True)
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

if __name__ == "__main__":
    mode = "train"
    type = "DQN"
    if mode == "train":
        train(type)
    else:
        test(type)
