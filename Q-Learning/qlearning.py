import gym
import numpy as np
from matplotlib import pyplot as plt


def state_to_disc(state):
    disc_state = []
    for i in range(len(state)):
        disc_state.append(np.abs(value_grids[i] - state[i]).argmin())
    return tuple(disc_state)

# Use LunarLander-v2 for the second part.
env = gym.make('CartPole-v0')

episodes = 20000
test_episodes = 10
action_dim = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.99
alpha = 0.1
a = 1000

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

value_grids = (x_grid, v_grid, th_grid, av_grid)

# Table for Q values
q_grid = np.zeros((discr, discr, discr, discr, action_dim))

epsilon = 0.4
# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    # Do a greedy test run in the last few episodes and render it
    test = ep > episodes

    # Initialize things
    state, done, steps = env.reset(), False, 0
    
    #epsilon = a / (a+ep)
    
    # Loop through the episode
    while not done:
        # Pick a random action (change it!)
        
        disc_vals = state_to_disc(state)
        if (np.random.random() < epsilon):
            action = int(np.random.random()*action_dim)
        else:
            act_0 = q_grid[disc_vals + (0,)]
            act_1 = q_grid[disc_vals + (1,)]             
            action = np.array([act_0, act_1]).argmax()
            #action = q_grid[disc_vals].argmax

        Q_curr =  q_grid[disc_vals + (action,)]

        # Perform the action
        state, reward, done, _ = env.step(action)
        disc_vals_next = state_to_disc(state)
        act_0 = q_grid[disc_vals_next + (0,)]
        act_1 = q_grid[disc_vals_next + (1,)]
        Q_next = max([act_0, act_1])
        
        q_grid[disc_vals + (action,)] = Q_curr + alpha*(reward + gamma*Q_next - Q_curr)

        # Draw if testing
        if test:
            env.render()

        steps += 1

    # Bookkeeping for plots
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[min(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[min(0, ep-200):])))


# Save the Q-value array
np.save("q_values.npy", q_grid)

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
