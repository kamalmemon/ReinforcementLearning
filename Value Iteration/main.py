import numpy as np
from time import sleep
from sailing import SailingGridworld
import random


gamma = 0.9

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
state = env.reset()

# TODO: Compute value function and policy.
value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h)),

def action_values(state, V):
    A = []
    for transition in env.transitions[state[0], state[1]]:
        action_value = 0
        for next_state, reward, done, prob in transition:
            action_value += prob * (reward + (gamma * 
                                    V[next_state] if not done else 0))
        A.append(action_value)
    return A

old_values = value_est.copy()
eps = 0.0001

for lp in range(100):
    env.clear_text()
    for w in range(env.w):
        for h in range(env.h):
            A = action_values((w, h), value_est)
            value_est[w][h] = np.max(A)
            policy[w][h] = np.argmax(A)
    if ((value_est - old_values) < eps).all():
        print('Stopping')
        break
    else :
        old_values = np.copy(value_est)
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()

#These functions will show the value function and policy when the env is rendered.
# env.clear_text()
# env.draw_values(value_est)
# #env.draw_actions(policy)
# env.render()

# Wait a while before starting to move
discounted_reward = 0
i = 0
done = False
while not done:
    # TODO: Use the computed policy here.
    action = policy[state]
    state, reward, done, _ = env.step(action)
    discounted_reward += (gamma ** i) * reward
    i += 1
    env.render()
    sleep(0.5)
print(discounted_reward)
