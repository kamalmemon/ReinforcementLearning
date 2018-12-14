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

from envs import ObsNorm
from model import ActorCritic
from pong import Pong
from simple_ai import PongAi
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument("--load_checkpoint", default=False, help="Load the checkpoint and run from there")
parser.add_argument("--load_checkpoint_opponent", default=False, help="Load the checkpoint and run from there")

def load_checkpoint(model, filename='/output/checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(args):
    env = Pong(headless= False)
    
    model = ActorCritic(1, 3)
    opponent = ActorCritic(1, 3)
    
    opponent.eval()
    model.eval()
    obsNorm = ObsNorm();
    obsNorm_op = ObsNorm();
    
    env.set_names('Player 1', 'Player 2')
    state = obsNorm.prepro(env.reset()[0])
    
    state = torch.from_numpy(state)
    state_op = state
    reward_sum = 0
    done = True
    
    done_count = 0
    actions = []

    if args.load_checkpoint:
        load_checkpoint(model, args.load_checkpoint)
        
    if args.load_checkpoint_opponent:

        load_checkpoint(opponent, args.load_checkpoint_opponent)
        opponent.eval()
        
    while True:
        env.render()
        # Sync with the shared model
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            
            cx1 = torch.zeros(1, 256)
            hx1 = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
            
            cx1 = cx.detach()
            hx1 = hx.detach()
    
        with torch.no_grad():
            inputTensor = state.unsqueeze(0);
            inputTensor_op = state_op.unsqueeze(0);
            
            value, logit, (hx, cx) = model((inputTensor, (hx, cx)))
            value_op, logit_op, (hx1, cx1) = opponent((inputTensor_op, (hx1, cx1)))
            
        prob = F.softmax(logit, dim=-1)
        prob_op = F.softmax(logit_op, dim=-1)
        #print(prob)
        action = prob.max(1, keepdim=True)[1].numpy()
        action2 = prob_op.max(1, keepdim=True)[1].numpy()
        
        (state, state_op), (reward, reward2), done, info = env.step((action[0,0], action2[0,0]))
        
        
        done = done or done_count >= 500
        reward_sum += reward
        state = obsNorm.prepro(state)
        state_op = obsNorm_op.prepro(state_op)
        
        #print(state.squeeze(0).shape)
        #plt.imshow(state.squeeze(0))
        #plt.show()
        
        #actions.append(actions[0]) 
        
        #if actions.count(actions[0]) == 5000:
         #   done = True
        if done:
           actions.clear()
           state = obsNorm.prepro(env.reset()[0])
           obsNorm.reset()
           state_op = state
           obsNorm_op.reset()
           done_count +=1
        if done_count == 50:
            break;
        
        state = torch.from_numpy(state)
        state_op = torch.from_numpy(state_op)
            
if __name__ == "__main__":           
    args = parser.parse_args()
    train(args)
