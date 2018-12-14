#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:48:34 2018

@author: daniyalusmani
"""
from PIL import Image
import argparse
import pickle
import numpy as np
import gym
import matplotlib.image as mpimg
from simple_ai import PongAi
from pong import Pong
from agent import Policy, Agent
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument('--hidden_layer_size', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size_episodes', type=int, default=1)
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

env = Pong(headless= args.headless)

player_id = 1
opponent_id = 3 - player_id
action_space = 1

UP_ACTION = 1
DOWN_ACTION = 2

episode_n = 1500

# Policy declaration
policy1 = Policy(args.hidden_layer_size, action_space)
policy2 = Policy(args.hidden_layer_size, action_space)

# Agent declaration
#opponent = Agent(env,  policy1, args.learning_rate, args.discount_factor, opponent_id, 'player 2')
opponent = PongAi(env, opponent_id)
player = Agent(env,  policy2, args.learning_rate, args.discount_factor, player_id, 'player 1')
player.load_checkpoint('checkpoint-single-2/checkpoints-player-1500.pth')
env.set_names(player.get_name(), opponent.get_name())


# action1 = player.get_action()
# action2 = opponent.get_action()
# (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

def prepro(I):
    """ prepro 210x210x3 uint8 frame into 6400 (80x80) 1D float vector """
    I[I != 0] = 255
    I = I[::2, ::2, :]  # downsample by factor of 2
    im = Image.fromarray(I)
    im = im.convert('L'); # convert to gray scale
    bw = np.asarray(im).copy() # copy to numpy array
    bw[bw != 0] = 1    # black and white
    return bw.astype(np.float).ravel() # into single array

while episode_n < 20000:
    print("Starting episode %d" % episode_n)

    episode_done = False
    
    episode_reward_sum_player_1 = 0
    episode_reward_sum_player_2 = 0
    
    smooth_reward_player_1 = 0
    smooth_reward_player_2 = 0
    
    
    env.reset()

    # First action
    action1 = np.random.choice(1,[0,1,2])
    #action2 = np.random.choice(1,[0,1,2])
    action2 = opponent.get_action()

    (obs1, obs2), _, _, _ = env.step((action1, action2))
    last_observation_player1 = prepro(obs1)
    last_observation_player2 = prepro(obs2)
    
    # Second action
    action1 = np.random.choice(1,[0,1,2])
    action2 = opponent.get_action()

    (obs1, obs2), _, _, _ = env.step((action1, action2))
    observation_player_1 = prepro(obs1)
    observation_player_2 = prepro(obs2)
    
    round_n = 1 
    n_steps = 1
    
    episode_done = 0

    while episode_done < 20:
        if args.render:
            env.render()

        observation_delta_player_1 = observation_player_1 - last_observation_player1
        observation_delta_player_2 = observation_player_2 - last_observation_player2
        
        last_observation_player_1 = observation_player_1
        last_observation_player_2 = observation_player_2
        
        up_probability_player_1 = player.get_action(observation_delta_player_1)
        action2 = opponent.get_action()
        
        if np.random.uniform() < up_probability_player_1:
            action1 = env.MOVE_UP
        else:
            action1 = env.MOVE_DOWN
        
        #if np.random.uniform() < up_probability_player_2:
         #   action2 = UP_ACTION
       # else:
        #    action2 = DOWN_ACTION

        (obs1, obs2), (reward1, reward2), done, info = env.step((action1, action2))
        observation_player_1 = prepro(obs1)
        observation_player_2 = prepro(obs2)

        episode_reward_sum_player_1 += reward1
        episode_reward_sum_player_2 += reward2
        
        n_steps += 1

        player.store_outcome(observation_player_1, action1, up_probability_player_1, reward1)
        #opponent.store_outcome(observation_player_2, action2, up_probability_player_2, reward2)

        if reward1 == +10:
            print("Round %d: %d time steps; player 1 won!" % (round_n, n_steps))
        elif reward2 == +10:
            print("Round %d: %d time steps; player 2 won!" % (round_n, n_steps))
            
        if reward1 != 0 or reward2 != 0:
            round_n += 1
            n_steps = 0
            env.reset()
        if done == True:
            episode_done += 1

    print("Episode %d finished after %d rounds" % (episode_n, round_n))

    # exponentially smoothed version of reward
    # if smoothed_reward is None:
     #   smooth_reward_player_1 = episode_reward_sum_player_1
      #  smooth_reward_player_2 = episode_reward_sum_player_2
    #else:
    smooth_reward_player_1 = smooth_reward_player_1 * 0.99 + episode_reward_sum_player_1 * 0.01
    smooth_reward_player_2 = smooth_reward_player_2 * 0.99 + episode_reward_sum_player_2 * 0.01
        
    print("Reward total for player 1 was %.3f; discounted moving average of reward is %.3f" \
        % (episode_reward_sum_player_1, smooth_reward_player_1))
    print("Reward total for player 2 was %.3f; discounted moving average of reward is %.3f" \
        % (episode_reward_sum_player_2, smooth_reward_player_2))

    if episode_n % args.batch_size_episodes == 0:
        player.episode_finish()
        #opponent.episode_finish()
    if episode_n % 500 == 0:
       player.save_checkpoint('checkpoint-single-2/checkpoints-player-%i.pth'%(episode_n));
       # opponent.save_checkpoint('checkpoints-3/checkpoints-opponent-%i.pth'%(episode_n));

   # if episode_n % args.checkpoint_every_n_episodes == 0:
   #     network.save_checkpoint()

    episode_n += 1