import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from ac_agent import Agent, Policy, Value
from cp_cont import CartPoleEnv
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
parser.add_argument("--train_episodes", type=int, default=20000, help="Number of episodes to train for")
parser.add_argument("--render_training", action='store_true',
                    help="Render each frame during training. Will be slower.")
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
            obsv_ = observation
            action, log_prob = agent.get_action(observation)

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action.detach().numpy())
            
            #storing observation
            agent.store_outcome(reward, log_prob, obsv_)

            # Draw the frame, if desired
            if args.render_training:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

            # #q4
            # if timesteps % 10 == 0:
            #     agent.episode_finished(episode_number)

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

        # If we managed to stay alive for 30 full episodes, assume it's learned
        # (in the default setting)
        if np.mean(timestep_history[-30:]) == env._max_episode_steps:
            print("Looks like it's learned. Finishing up early")
            break
        
        # #q4
        # if timesteps % 10 > 1:
        #     agent.episode_finished(episode_number)

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)
        

    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    plt.title("Reward history (sig=%f)" % agent.sigma)
    plt.show()
    print("Training finished.")
    plot_heatmaps()
    

# Create a Gym environment
env = CartPoleEnv()

# For CartPole - maximum episode length
env._max_episode_steps = 1000

# Get dimensionalities of actions and observations
action_space_dim = 1
observation_space_dim = 4

# Create the agent, value estimates and the policy
policy = Policy(observation_space_dim)
value_nn = Value(observation_space_dim)
agent = Agent(policy, value_nn)

def plot_heatmaps():
    xspace = np.linspace(-2.4, 2.4, 40)
    tspace = np.linspace(-0.3, 0.3, 40)

    val_estimates = np.zeros((40,40))

    i = 0
    for x in xspace:
        j = 0
        for t in tspace:
            state = torch.from_numpy(np.array([x, 0, t, 0])).float()
            val_estimates[i, j] = agent.value.forward(state)
            j += 1
        i += 1
    pickle.dump(val_estimates,open('heatmap2.p','wb'))
    
    plt.imshow(val_estimates.T)
    plt.colorbar()
    plt.show()


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


