#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:48:34 2018

@author: daniyalusmani
"""

from PIL import Image
from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
import argparse
from agent import Policy, Agent

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()


env = Pong(headless= args.headless)

UP_ACTION = 1
DOWN_ACTION = 2

# Policy declaration
policy1 = Policy(500, 1)
policy2 = Policy(500, 1)

# Agent declaration
opponent = Agent(env,  policy2, 0.0005, 0.99, 2, 'player 1')
opponent = PongAi(env, 2)
player = Agent(env,  policy1, 0.0005, 0.99, 1, 'player 2')

# print(player.policy.state_dict())
player.load_checkpoint('checkpoint-single-1/checkpoints-player-18000.pth')
#player.load_checkpoint('checkpoints-3/checkpoints-player-9500.pth')
#opponent.load_checkpoint('checkpoints-3/checkpoints-opponent-9500.pth')
# player.policy.state_dict()

def plot(observation):
    plt.imshow(observation/255)
    plt.show()

episodes = 20

player_id = 1
opponent_id = 3 - player_id
#opponent = PongAi(env, opponent_id)
#player = PongAi(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

def prepro(I):
    """ prepro 210x210x3 uint8 frame into 6400 (80x80) 1D float vector """
    I[I != 0] = 255
    I = I[::2, ::2, :]  # downsample by factor of 2
    im = Image.fromarray(I)
    im = im.convert('L'); # convert to gray scale
    bw = np.asarray(im).copy() # copy to numpy array
    bw[bw != 0] = 1    # black and white
    return bw.astype(np.float).ravel() # into single array

for i in range(0,episodes):
    observation = prepro(env.reset()[0])
    action1 = np.random.choice(1,[0,1,2])
    action2 = np.random.choice(1,[0,1,2])
    
    (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
    
    obs_player_1 = prepro(ob1)
    obs_player_2 = prepro(ob2)
    
    diff_obs_player_1 = obs_player_1 - observation
    diff_obs_player_2 = obs_player_2 - observation
    done = False
    while not done:
         
        up_probability_player_1 = player.get_action(diff_obs_player_1)
        action2 = opponent.get_action()
        # up_probability_player_2 = opponent.get_action(diff_obs_player_2)
        
        if np.random.uniform() < up_probability_player_1:
            action1 = UP_ACTION
        else:
            action1 = DOWN_ACTION
            
        #if np.random.uniform() < up_probability_player_2:
         #   action2 = UP_ACTION
        #else:
         #   action2 = DOWN_ACTION


                
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        prev_obs_player_1 = obs_player_1
        prev_obs_player_2 = obs_player_2
        
        obs_player_1 = prepro(ob1)
        obs_player_2 = prepro(ob2)
        
        diff_obs_player_1 = obs_player_1 - prev_obs_player_1
        diff_obs_player_2 = obs_player_2 - prev_obs_player_2
                
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(i))

# Needs to be called in the end to shut down pygame
env.end()



