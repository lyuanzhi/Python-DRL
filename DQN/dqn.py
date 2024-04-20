import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x


class DQN(object):
    def __init__(self, n_states, n_actions, hidden_dim=64, critic_lr=1e-3, gamma=0.9, explore_intensity=100, replay_capacity=1000, replay_batch_size=32, target_update_freq=20,
                  is_double_dqn=True, is_load=False, critic_path_name="dqn_critic_net.pkl", is_train=True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.MEMORY_CAPACITY = replay_capacity
        self.BATCH_SIZE = replay_batch_size
        self.gamma = gamma
        self.TARGET_NETWORK_REPLACE_FREQ = target_update_freq
        self.is_double_dqn = is_double_dqn
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.learn_time = 1.0
        self.add_learn_time = 1.0 / explore_intensity
        if not is_train:
            self.learn_time = 100.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_load:
            self.critic = torch.load(critic_path_name)
        else:
            self.critic = Critic(self.n_states, self.n_actions, hidden_dim).to(self.device)

        self.critic_target = Critic(self.n_states, self.n_actions, hidden_dim).to(self.device)
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss_func = nn.MSELoss()

    def select_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s).to(self.device), 0)
        if np.random.uniform() < (1 - 0.5 ** self.learn_time):
            actions_value = self.critic(s)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy().squeeze(0)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_data(self, s, a, r, s_, d):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if d:
            self.learn_time += self.add_learn_time

    def learn(self):
        if self.memory_counter > self.MEMORY_CAPACITY:
            if self.learn_step_counter % self.TARGET_NETWORK_REPLACE_FREQ == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
            self.learn_step_counter += 1

            sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)

            b_memory = self.memory[sample_index, :]
            b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states])).to(self.device)
            b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))).to(self.device)
            b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])).to(self.device)
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:])).to(self.device)

            q_eval = self.critic(b_s).gather(1, b_a)
            if self.is_double_dqn:
                a_next_arg = self.critic(b_s_).detach().argmax(1).reshape(self.BATCH_SIZE, 1)  # Double DQN
                q_next = self.critic_target(b_s_).detach().gather(1, a_next_arg)  # Double DQN
                q_target = b_r + self.gamma * q_next  # Double DQN
            else:
                q_next = self.critic_target(b_s_).detach()  # DQN
                q_target = b_r + self.gamma * q_next.max(1)[0].view(self.BATCH_SIZE, 1)  # DQN
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save(self, critic_path_name="dqn_critic_net.pkl"):
        torch.save(self.critic, critic_path_name)
