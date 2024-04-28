# import gymnasium
# env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode='human')

# env.reset()

# episodes = 500
# while episodes > 0:
#     done = False
#     observation, reward, done, truncated, info = env.step(env.action_space.sample())
#     env.render()

#     if not done:
#         episodes -= 1
    
# env.close()


import gymnasium
import torch
import random
import numpy as np

env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode='human')

def Create_Random_Trajectories(num_trajectories: int, steps: int):
    """Creates specified number of random trajectories for BipedalWalker-v3

    Args:
        num_trajectories (int): number of trajectories to create
        steps (int): how many steps to take per episode

    Returns:
        list[np.array]: dimensions are [number of steps before episode terminated, size of state space]
    """
    
    action_size = env.action_space.shape[0]
    trajectories = []
    cum_rews = []
    while num_trajectories > 0:
        time_steps = steps
        env.reset()
        done = False
        trajectory = []
        cum_reward = 0
        while time_steps > 0 and not done:
            #done = False
            env.render()
            action = np.random.uniform(-1.0, 1.0, size=action_size)
            next_state, reward, done, truncated, info = env.step(action)
            cum_reward += reward
            time_steps -= 1
            trajectory.append(np.array(next_state))
        trajectory = np.array(trajectory)
        num_trajectories -= 1
        trajectories.append(trajectory)
        cum_rews.append(cum_reward)
    return trajectories, cum_rews

                
rand_trajs = Create_Random_Trajectories(2, 100)