import gym
import numpy as np
from matplotlib import pyplot as plt

def state_to_disc(state):
    x_disc = np.abs(x_grid - state[0]).argmin()
    y_disc = np.abs(y_grid - state[1]).argmin()
    v_disc = np.abs(v_grid - state[2]).argmin()
    v2_disc = np.abs(v2_grid - state[3]).argmin()
    th_disc = np.abs(th_grid - state[4]).argmin()
    av_disc = np.abs(av_grid - state[5]).argmin()
    return (x_disc, y_disc, v_disc, v2_disc, th_disc, av_disc, int(state[6]), int(state[7]))

# Use LunarLander-v2 for the second part.
env = gym.make('LunarLander-v2')

episodes = 20#000
test_episodes = 10
action_dim = 4

discr = 16
x_min, x_max = -2.4, 2.4
y_min, y_max = -2.4, 2.4
v_min, v_max = -3, 3
v2_min, v2_max = -3, 3
th_min, th_max = -3, 3
av_min, av_max = -4, 4

# Parameters
gamma = 0.99
alpha = 0.1
a = 1000

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
v2_grid = np.linspace(v2_min, v2_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Table for Q values
q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, action_dim))

# Training loop
ep_lengths, epl_avg = [], []
reward_history, average_reward_history = [], []
for ep in range(episodes+test_episodes):
    # Do a greedy test run in the last few episodes and render it
    test = ep > episodes
    reward_sum = 0

    # Initialize things
    state, done, steps = env.reset(), False, 0

    # Loop through the episode
    while not done:
        # Pick a random action (change it!)
        epsilon = a / (a+ep)
        disc_vals = state_to_disc(state)
        if (np.random.random() < epsilon):
            action = int(np.random.random()*action_dim)
        else:
            act_0 = q_grid[disc_vals + (0,)]
            act_1 = q_grid[disc_vals + (1,)]
            act_2 = q_grid[disc_vals + (2,)]
            act_3 = q_grid[disc_vals + (3,)]             
            action = np.array([act_0, act_1, act_2, act_3]).argmax()

        Q_curr =  q_grid[disc_vals + (action,)]

        # Perform the action
        state, reward, done, _ = env.step(action)
        disc_vals_next = state_to_disc(state)
        act_0 = q_grid[disc_vals_next + (0,)]
        act_1 = q_grid[disc_vals_next + (1,)]
        act_2 = q_grid[disc_vals_next + (2,)]
        act_3 = q_grid[disc_vals_next + (3,)] 
        Q_next = max([act_0, act_1, act_2, act_3])
        
        q_grid[disc_vals + (action,)] = Q_curr + alpha*(reward + gamma*Q_next - Q_curr)

        # Draw if testing
        if test:
            env.render()
        else:
            reward_sum += reward
        steps += 1
    if not test:
        reward_history.append(reward_sum)
        if ep > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

    # Bookkeeping for plots
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[min(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[min(0, ep-200):])))


# Save the Q-value array
np.save("q_values.npy", q_grid)

plt.plot(reward_history)
plt.plot(average_reward_history)
plt.legend(["Reward", "100-episode average"])
plt.show()

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
