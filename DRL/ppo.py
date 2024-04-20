import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()

        self.fc = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, softmax_dim):
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=softmax_dim)
        return x


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()

        self.fc = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x


class PPO(object):
    def __init__(self, n_states, n_actions, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2, learn_epochs=20, is_load=False, 
                 actor_path_name="ppo_actor_net.pkl", critic_path_name="ppo_critic_net.pkl", is_train=True):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.learn_epochs = learn_epochs
        self.eps_clip = eps_clip
        self.eps = np.finfo(np.float32).eps.item()
        self.data = []
        self.oneTimeProb = 0
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_load:
            self.actor = torch.load(actor_path_name)
            self.critic = torch.load(critic_path_name)
        else:
            self.actor = Actor(n_states, n_actions, hidden_dim).to(self.device)
            self.critic = Critic(n_states, hidden_dim).to(self.device)
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': actor_lr},
                                     {'params': self.critic.parameters(), 'lr': critic_lr}])
        self.MseLoss = nn.MSELoss()

    def select_action(self, s):
        with torch.no_grad():
            s = torch.FloatTensor(s).to(self.device)
            prob = self.actor(s, softmax_dim=0)
            if not self.is_train:
                return torch.argmax(prob).data.cpu().numpy()
            m = Categorical(prob)
            a = m.sample()
            a = a.item()
            self.oneTimeProb = prob[a].item()
        return a

    def store_data(self, s, a, r, s_, d):
        self.data.append((s, a, r, self.oneTimeProb, d))

    def make_batch(self):
        s_lst, a_lst, r_lst, done_lst, prob_a_lst = [], [], [], [], []
        for t in self.data:
            s, a, r, prob_a, done = t
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            done_mask = 0 if done else 1
            done_lst.append(done_mask)
            prob_a_lst.append([prob_a])

        self.data = []
        return s_lst, a_lst, r_lst, done_lst, prob_a_lst

    def learn(self):
        s_lst, a_lst, r_lst, done_lst, prob_a_lst = self.make_batch()

        s = torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device)
        a = torch.tensor(np.array(a_lst)).long().to(self.device)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float).to(self.device)

        for i in range(self.learn_epochs):
            rewards = []
            discounted_reward = 0.0
            for r, done in zip(reversed(r_lst), reversed(done_lst)):
                discounted_reward = r + self.gamma * discounted_reward * done
                rewards.append([discounted_reward])
            rewards.reverse()
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

            state_value = self.critic(s)
            advantage = rewards - state_value.detach()

            prob = self.actor(s, softmax_dim=1)
            dist = Categorical(prob)
            dist_entropy = dist.entropy()
            actor_a = prob.gather(1, a)
            ratio = torch.exp(torch.log(actor_a) - torch.log(prob_a))

            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            loss = -torch.min(surr1, surr2) + self.MseLoss(self.critic(s), rewards) / (rewards.std() + self.eps) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save(self, actor_path_name="ppo_actor_net.pkl", critic_path_name="ppo_critic_net.pkl"):
        torch.save(self.actor, actor_path_name)
        torch.save(self.critic, critic_path_name)
