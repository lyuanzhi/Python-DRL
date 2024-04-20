import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        pi = dist.rsample()
        log_pi = dist.log_prob(pi)
        pi = torch.tanh(pi)
        log_pi = (log_pi - torch.log(1 - pi.pow(2) + 1e-6)).sum(dim=-1)
        pi = pi * self.max_action
        return pi, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device):
        self.s_buf = np.zeros([max_size, state_dim], dtype=np.float32)
        self.ns_buf = np.zeros([max_size, state_dim], dtype=np.float32)
        self.a_buf = np.zeros([max_size, action_dim], dtype=np.float32)
        self.r_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size
        self.device = device

    def add(self, s, a, r, ns, done):
        self.s_buf[self.ptr] = s
        self.ns_buf[self.ptr] = ns
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        return dict(s=torch.Tensor(self.s_buf[index]).to(self.device),
                    ns=torch.Tensor(self.ns_buf[index]).to(self.device),
                    a=torch.Tensor(self.a_buf[index]).to(self.device),
                    r=torch.Tensor(self.r_buf[index]).to(self.device),
                    done=torch.Tensor(self.done_buf[index]).to(self.device))


class SAC(object):
    def __init__(self, n_states, n_actions, max_action, hidden_dim=300, actor_lr=3e-3, critic_lr=3e-3, alpha_lr=3e-3, alpha=0.5, gamma=0.99, tau=0.005, replay_capacity=10000, 
                 replay_batch_size=100, automatic_entropy_tuning=True, is_load=False, actor_path_name="sac_actor_net.pkl", is_train=True):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = replay_batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_load:
            self.actor = torch.load(actor_path_name)
        else:
            self.actor = Actor(n_states, n_actions, max_action, hidden_dim).to(self.device)

        self.critic1 = Critic(n_states + n_actions, hidden_dim).to(self.device)
        self.critic2 = Critic(n_states + n_actions, hidden_dim).to(self.device)
        self.critic1_target = Critic(n_states + n_actions, hidden_dim).to(self.device)
        self.critic2_target = Critic(n_states + n_actions, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=critic_lr)
        self.replay_buffer = ReplayBuffer(n_states, n_actions, replay_capacity, self.device)

        if self.automatic_entropy_tuning:
            self.target_entropy = -n_actions
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, s):
        a, _ = self.actor(torch.Tensor(s).to(self.device))
        a = a.detach().cpu().numpy()
        return a

    def store_data(self, s, a, r, s_, d):
        self.replay_buffer.add(s, a, r, s_, d)

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size)
        s = batch['s']
        ns = batch['ns']
        a = batch['a']
        r = batch['r']
        done = batch['done']

        next_pi, next_log_pi = self.actor(ns)
        target_Q = torch.min(self.critic1_target(ns, next_pi), self.critic2_target(ns, next_pi)).squeeze(1)
        target_Q = r + self.gamma * (1 - done) * (target_Q - self.alpha * next_log_pi)
        current_Q1 = self.critic1(s, a).squeeze(1)
        current_Q2 = self.critic2(s, a).squeeze(1)
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor(s)
        actor_loss = torch.min(self.critic1(s, pi), self.critic2(s, pi)).squeeze(1)
        actor_loss = (self.alpha * log_pi - actor_loss).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for main_param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

        for main_param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, actor_path_name="sac_actor_net.pkl"):
        torch.save(self.actor, actor_path_name)
