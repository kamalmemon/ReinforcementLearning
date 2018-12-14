import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards, softmax_sample


class Value(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.fc1 = torch.nn.Linear(state_space, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        return self.fc3(l2)

class Policy(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.fc1 = torch.nn.Linear(state_space, 30)
        self.fc_mean = torch.nn.Linear(30, 1)
        self.fc_var = torch.nn.Linear(30, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        xmean = self.fc_mean(x)
        xvar = self.fc_var(x)
        return F.sigmoid(xmean), F.sigmoid(xvar)



class Agent(object):
    def __init__(self, policy, value):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.value = value.to(self.train_device)
        self.optimizer_p = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.optimizer_v = torch.optim.RMSprop(value.parameters(), lr=5e-3)
        # self.batch_size = 1
        self.gamma = 0.98
        self.observations = []
        self.actions = []
        self.rewards = []
        self.sigma = 0.5
        self.episode_number = 0
        self.values = []

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)
        
        self.values, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)

        error = discounted_rewards - all_values
        error -= torch.mean(error)
        error /= torch.std(error.detach())

        self.optimizer_p.zero_grad()
        self.optimizer_v.zero_grad()

        
        p_loss = (error.detach() * all_actions).sum()
        c_loss = error.pow(2).mean()
        
        p_loss.backward()
        c_loss.backward()
        
        self.optimizer_p.step()
        self.optimizer_v.step()
        

        self.episode_number += 1

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # constant sigma
        # mean_val = torch.tensor(self.policy.forward(x))
        # variance = torch.tensor(self.sigma)

        #decaying sigma
        # mean_val = torch.tensor(self.policy.forward(x))
        # variance = torch.tensor(1000/(1000+self.episode_number))

        # sigma from nn
        mean_val, variance = self.policy.forward(x)

        normal_dist = Normal(torch.tensor(mean_val), torch.tensor(variance))
        sample_val = normal_dist.sample()
        sample_val_norm = sample_val*50 - 25
        sample_log_prob = normal_dist.log_prob(sample_val)

        return sample_val_norm, -sample_log_prob

    def store_outcome(self, reward, log_action_prob, observation):
        x = torch.from_numpy(observation).float().to(self.train_device)

        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.values.append(self.value.forward(x))