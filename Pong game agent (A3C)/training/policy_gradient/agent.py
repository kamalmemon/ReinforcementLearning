#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:41:54 2018

@author: daniyalusmani
"""
import torch
import torch.nn.functional as F
import numpy as np;
from pong import Pong

OBSERVATIONS_SIZE = 10500

class Policy(torch.nn.Module):
    def __init__(self, hidden_layer_size, action_space):
        super().__init__()
        # Create layers etc
        self.hidden_layer_size = hidden_layer_size
        self.action_space = action_space 
        
        self.fc1 = torch.nn.Linear(OBSERVATIONS_SIZE, self.hidden_layer_size)
        self.fc2 = torch.nn.Linear(self.hidden_layer_size, action_space)
      
        # Initialize neural network weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)       
        return F.sigmoid(x);



class Agent(object):
    def __init__(self, env,  policy, learning_rate, discount_factor, player_id=1, name = 'player1'):
        if type(env) is not Pong:
            raise TypeError("Policy gradient AI")
            
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        
        self.sampled_actions = []
        self.rewards = []
        self.observation = []
        self.up_probability = []
        
        self.discount_factor = discount_factor;
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.learning_rate)
        self.player_id = player_id
        self.bpe = 4
        self.name = name


    def get_name(self):
        return self.name


    def discount_rewards(self, r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


    def get_action(self, observation = []):
        if len(observation) == 0:
            return np.random.choice(1,[0,1])
        else:
            return self.policy.forward(observation)
    
    def episode_finish(self):
        all_actions = torch.stack(self.sampled_actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_up_probability = torch.stack(self.up_probability, dim=0).to(self.train_device).squeeze(-1)
        self.up_probability, self.rewards, self.sampled, self.sampled_actions = [], [], [], []
        
        discounted_rewards = self.discount_rewards(all_rewards, self.discount_factor)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        self.loss_fn = torch.nn.BCELoss(reduction='sum', weight = discounted_rewards)
        all_actions = torch.tensor(all_actions,requires_grad=True)
        
        loss = self.loss_fn(all_actions, all_up_probability)
        self.optimize(loss)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def store_outcome(self, observation, action, up_prob, reward):
        if  int(action) == 1:
            self.sampled_actions.append(torch.Tensor([action]));
        else:
            # for down
            self.sampled_actions.append(torch.Tensor([0.0]));

        self.rewards.append(torch.Tensor([reward]))
        # self.rewards.append(torch.Tensor([observation]))
        self.up_probability.append(torch.Tensor([up_prob]))
        
    def reset(self):
        # Nothing to done for now...
        return

    def save_checkpoint(self, filename='/output/checkpoint.pth.tar'):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
                }, filename)
        
    def load_checkpoint(self, filename='/output/checkpoint.pth.tar'):
        checkpoint = torch.load(filename)
        print(checkpoint)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

