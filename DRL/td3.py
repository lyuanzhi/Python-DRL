import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim):
        super(Actor, self).__init__()

        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.max_action * torch.tanh(self.f3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.c1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.c2 = nn.Linear(hidden_dim, hidden_dim)
        self.c3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x1 = F.relu(self.c1(torch.cat([x, u], 1)))
        x1 = F.relu(self.c2(x1))
        x1 = self.c3(x1)

        x2 = F.relu(self.d1(torch.cat([x, u], 1)))
        x2 = F.relu(self.d2(x2))
        x2 = self.d3(x2)
        return x1, x2


class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.storage = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self):
        ind = np.random.randint(0, len(self.storage), size=self.batch_size)
        state, action, next_state, reward, done = [], [], [], [], []
        for i in ind:
            s, a, s_, r, d = self.storage[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            next_state.append(np.array(s_, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))
        return np.array(state), np.array(action), np.array(next_state), np.array(reward).reshape(-1, 1), np.array(
            done).reshape(-1, 1)


class TD3(object):
    def __init__(self, n_states, n_actions, max_action, hidden_dim=300, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.0005, learn_epochs=200, replay_capacity=1000000, 
                 replay_batch_size=100, is_load=False, actor_path_name="td3_actor_net.pkl", critic_path_name="td3_critic_net.pkl", is_train=True):
        self.tau = tau
        self.gamma = gamma
        self.learn_epochs = learn_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_load:
            self.actor = torch.load(actor_path_name)
            self.critic = torch.load(critic_path_name)
        else:
            self.actor = Actor(n_states, n_actions, max_action, hidden_dim).to(self.device)
            self.critic = Critic(n_states, n_actions, hidden_dim).to(self.device)

        self.actor_target = Actor(n_states, n_actions, max_action, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_target = Critic(n_states, n_actions, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(replay_capacity, replay_batch_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.squeeze(0).cpu().detach().numpy()
        return action
    
    def store_data(self, s, a, r, s_, d):
        self.replay_buffer.push((s, a, s_, r, float(d)))

    def learn(self):
        for i in range(self.learn_epochs):
            s, a, s_, r, d = self.replay_buffer.sample()
            state = torch.FloatTensor(s).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            next_state = torch.FloatTensor(s_).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)

            a_ = self.actor_target(next_state)
            # policy smoothing
            policy_noise = (torch.rand_like(a_) * 0.2).clamp(-0.5, 0.5)
            a_ = (a_ + policy_noise).clamp(-2, 2)
            q1, q2 = self.critic_target(next_state, a_)
            target_Q = torch.min(q1, q2)
            target_Q = reward + (done * self.gamma * target_Q).detach()
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed update
            if i % 2 == 0:
                actor_loss, _ = self.critic(state, self.actor(state))
                actor_loss = -actor_loss.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, actor_path_name="td3_actor_net.pkl", critic_path_name="td3_critic_net.pkl"):
        torch.save(self.actor, actor_path_name)
        torch.save(self.critic, critic_path_name)
