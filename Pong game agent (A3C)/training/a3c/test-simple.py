#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:57:21 2018

@author: daniyalusmani
"""
import time
from collections import deque
import argparse
import torch
import torch.nn.functional as F

from pong import Pong
from simple_ai import PongAi
import matplotlib.pyplot as plt
from agent import Agent

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument("--load_checkpoint", default=False, help="Load the checkpoint and run from there")

def load_checkpoint(model, filename='/output/checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(args):
    env = Pong(headless= False)
    
    #model = ActorCritic(1, 3)
    
    opponent = PongAi(env, 2)
    agent = Agent();
    
    #model.eval()
    #obsNorm = ObsNorm();
    
    env.set_names(agent.get_name(), opponent.get_name())
    state = env.reset()[0]
    
    #state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    
    done_count = 0
    actions = []

    if args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
    
    while True:
        env.render()
        # Sync with the shared model
      #  if done:
      #      cx = torch.zeros(1, 256)
      #      hx = torch.zeros(1, 256)
      #  else:
       #     cx = cx.detach()
       #     hx = hx.detach()
    
       # with torch.no_grad():
       #     inputTensor = state.unsqueeze(0);
       #     value, logit, (hx, cx) = model((inputTensor, (hx, cx)))
       # prob = F.softmax(logit, dim=-1)
        #print(prob)
       # action = prob.max(1, keepdim=True)[1].numpy()
        action = agent.get_action(state)   
        action2 = opponent.get_action()
        
        (state, obs2), (reward, reward2), done, info = env.step((action, action2))
        
        
        done = done or done_count >= 500
        reward_sum += reward
        # state = obsNorm.prepro(state)
        
        #print(state.squeeze(0).shape)
        #plt.imshow(state.squeeze(0))
        #plt.show()
        
        actions.append(action)
        
        if actions.count(actions[0]) == 5000:
            done = True
        if done:
         #  actions.clear()
          # state = obsNorm.prepro(env.reset()[0])
         #  obsNorm.reset()
           state = env.reset()[0]
           agent.reset()
           done_count +=1
        if done_count == 50:
            break;
        
       # state = torch.from_numpy(state)
            
if __name__ == "__main__":           
    args = parser.parse_args()
    train(args)
