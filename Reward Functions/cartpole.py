import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from utils import get_space_dim

parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train for")
parser.add_argument("--render_training", action='store_true',
                    help="Render each frame during training. Will be slower.")
parser.add_argument("--render_test", action='store_true', help="Render test")
parser.add_argument("--x_0", type=float, default=0, help="An arbitrary point of the screen")
args = parser.parse_args()


# Policy training function
def train(train_episodes, agent):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            # Task 1 - change the reward function
            reward = new_reward(observation)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if args.render_training:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
              .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 10 full episodes, assume it's learned
        # (in the default setting)
        if np.mean(timestep_history[-10:]) == env._max_episode_steps:
            print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    plt.title("Reward history (%s)" % args.env)
    plt.show()
    print("Training finished.")


# Function to test a trained policy
def test(episodes, agent):
    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)
            # New reward function
            reward = new_reward(observation)
            if args.render_test:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

# Definition of the modified reward function
def new_reward(state):
    # #1
    #   return min(4, np.abs(1/state[0]))
    
    # #2
    # if state[0] == args.x_0:
    #     return 4
    # else:
    #     return min(4, 1/np.abs(state[0]-args.x_0))

    #3
    return 2+state[1]**2
    

    
    
    #return min(4, np.abs(1/state[0]))

# Create a Gym environment
env = gym.make(args.env)

# Exercise 1
# For CartPole-v0 - maximum episode length
env._max_episode_steps = 500

# Get dimensionalities of actions and observations
action_space_dim = get_space_dim(env.action_space)
observation_space_dim = get_space_dim(env.observation_space)

# Instantiate agent and its policy
policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy)

# Print some stuff
print("Environment:", args.env)
print("Training device:", agent.train_device)
print("Observation space dimensions:", observation_space_dim)
print("Action space dimensions:", action_space_dim)
print('Passed argument: ', args.x_0)

# If no model was passed, train a policy from scratch.
# Otherwise load the policy from the file and go directly to testing.
if args.test is None:
    try:
        train(args.train_episodes, agent)
    # Handle Ctrl+C - save model and go to tests
    except KeyboardInterrupt:
        print("Interrupted!")
    model_file = "%s_params.mdl" % args.env
    torch.save(policy.state_dict(), model_file)
    print("Model saved to", model_file)
else:
    state_dict = torch.load(args.test)
    policy.load_state_dict(state_dict)
print("Testing...")
test(100, agent)


