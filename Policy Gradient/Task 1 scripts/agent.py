import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.fc1 = torch.nn.Linear(state_space, 20)
        self.fc_mean = torch.nn.Linear(20, 1)
        self.fc_var = torch.nn.Linear(20, 1)
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
        return torch.sigmoid(xmean), torch.sigmoid(xvar)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        # self.batch_size = 1
        self.gamma = 0.98
        self.observations = []
        self.actions = []
        self.rewards = []
        self.sigma = 0.5
        self.episode_number = 0

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        #updating policy
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.episode_number += 1

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        #constant sigma
        
        # mean_val = torch.tensor(self.policy.forward(x))
        # variance = torch.tensor(self.sigma)

        #decaying sigma

        # mean_val = torch.tensor(self.)
        
        # mean_val = torch.tensor(self.policy.forward(x))
        # variance = torch.tensor(1000/(1000+self.episode_number))

        sigma from nn
        mean_val, variance = self.policy.forward(x)

        normal_dist = Normal(torch.tensor(mean_val), torch.tensor(variance))
        sample_val = normal_dist.sample()
        sample_val_norm = sample_val*50 - 25
        sample_log_prob = normal_dist.log_prob(sample_val)

        return sample_val_norm, -sample_log_prob

    def store_outcome(self, reward, log_action_prob):
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))