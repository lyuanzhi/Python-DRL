# Python-DRL
A pytorch-based deep reinforcement learning package. At present, I have implemented the following reinforcement learning algorithms: Deep Q-Network (DQN), Policy Gradient (PG), Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), and Soft Actor-Critic (SAC).

## Dependencies
```
numpy~=1.23.0
torch>=2.0.1
gym~=0.26.2
opencv-python>=4.5.5.62
pygame~=2.5.2
```

## Supported Python Versions
```
3.8
3.9
3.10
```

## Install
```
pip install Python-DRL
```

## Basic Usage Examples With Gym
```
python ./examples/cartpole-demo.py -m [train/test] -t [DQN/PPO/PG]
python ./examples/pendulum-demo.py -m [train/test] -t [DDPG/TD3/SAC]
```
##### Note that ```examples``` will not be installed by ```pip install Python-DRL```.

## Detailed Usage Guide
### DQN
```
from DRL.dqn import DQN

DQN(n_states, n_actions, hidden_dim=64, critic_lr=1e-3, gamma=0.9, explore_intensity=100, replay_capacity=1000, replay_batch_size=32, target_update_freq=20, is_double_dqn=True, is_load=False, critic_path_name="dqn_critic_net.pkl", is_train=True)
```

### PPO
```
from DRL.ppo import PPO

PPO(n_states, n_actions, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2, learn_epochs=20, is_load=False, actor_path_name="ppo_actor_net.pkl", critic_path_name="ppo_critic_net.pkl", is_train=True)
```

### PG
```
from DRL.pg import PG

PG(n_states, n_actions, hidden_dim=64, actor_lr=1e-3, gamma=0.99, is_load=False, actor_path_name="pg_actor_net.pkl", is_train=True)
```

### DDPG
```
from DRL.ddpg import DDPG

DDPG(n_states, n_actions, max_action, hidden_dim=300, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.0005, learn_epochs=200, replay_capacity=1000000, replay_batch_size=100, is_load=False, actor_path_name="ddpg_actor_net.pkl", critic_path_name="ddpg_critic_net.pkl", is_train=True)
```

### TD3
```
from DRL.td3 import TD3

TD3(n_states, n_actions, max_action, hidden_dim=300, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.0005, learn_epochs=200, replay_capacity=1000000, replay_batch_size=100, is_load=False, actor_path_name="td3_actor_net.pkl", critic_path_name="td3_critic_net.pkl", is_train=True)
```

### SAC
```
from DRL.sac import SAC

SAC(n_states, n_actions, max_action, hidden_dim=300, actor_lr=3e-3, critic_lr=3e-3, alpha_lr=3e-3, alpha=0.5, gamma=0.99, tau=0.005, replay_capacity=10000, replay_batch_size=100, automatic_entropy_tuning=True, is_load=False, actor_path_name="sac_actor_net.pkl", is_train=True)
```

## Updates Log
- 1.0.0
  - implemented DQN, PG, PPO, DDPG, TD3, and SAC (missing pygame)
- 1.0.1
  - add pygame dependency
