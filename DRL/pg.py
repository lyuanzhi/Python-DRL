import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()
        self.fc = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class PG(object):
    def __init__(self, n_states, n_actions, hidden_dim=64, actor_lr=1e-3, gamma=0.99, is_load=False, actor_path_name="pg_actor_net.pkl", is_train=True):
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.is_train = is_train
        self.eps = np.finfo(np.float32).eps.item()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_load:
            self.actor = torch.load(actor_path_name)
        else:
            self.actor = Actor(n_states, n_actions, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def select_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)
        probs = self.actor(state)
        if not self.is_train:
            return torch.argmax(probs).data.cpu().numpy()
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def store_data(self, s, a, r, s_, d):
        self.rewards.append(r)

    def learn(self):
        R = 0
        loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def save(self, actor_path_name="pg_actor_net.pkl"):
        torch.save(self.actor, actor_path_name)
