from DRL.ppo import PPO
from DRL.pg import PG
from DRL.dqn import DQN
import gym

env = gym.make('CartPole-v1', render_mode = "rgb_array")
EPOCH = 2
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

def train(type):
    if type == "DQN":
        agent = DQN(N_STATES, N_ACTIONS)
    if type == "PPO":
        agent = PPO(N_STATES, N_ACTIONS)
    if type == "PG":
        agent = PG(N_STATES, N_ACTIONS)
    ep_r = 0
    for i in range(EPOCH):
        s = env.reset()[0]
        while True:
            a = agent.select_action(s)
            s_, r, d, trunc, _ = env.step(a)
            if type == "DQN":
                r = (env.theta_threshold_radians - abs(s_[2])) / env.theta_threshold_radians * 0.7 + (env.x_threshold - abs(s_[0])) / env.x_threshold * 0.3
            agent.store_data(s, a, r, s_, d)
            s = s_
            ep_r += r
            if type == "DQN":
                agent.learn()
            if d or trunc:
                break
        if type == "PPO" or type == "PG":
            agent.learn()
        if i % 10 == 0 and i != 0:
            print("# episode :{}, avg score : {:.1f}".format(i, ep_r / 10))
            ep_r = 0
    agent.save()

def test_train():
    train("PPO")
