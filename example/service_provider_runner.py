import numpy as np
from amalearn.agent import ServiceProviderAgent
from amalearn.reward import GaussianReward
from amalearn.environment import MutliArmedBanditEnvironment

# The rewards and the environment are dummies because the agent know all the parameters of the world 
# and doesn't need the environment and the rewards to actually explore the environment

means = [1, 2, -10, -5]
stds = [0.2, 0.1, 0.5, 0.4]
rewards = [GaussianReward(mean, std) for mean, std in zip(means, stds)]
env = MutliArmedBanditEnvironment(rewards, 10, '1')
agent = ServiceProviderAgent('1', env)

while True:
    state_values, policy, policy_stable = agent.sweep()
    print('policy stable {}'.format(policy_stable))
    if policy_stable:
        break


np.save('policy_punish6.npy', policy)
np.save('state_values_punish6.npy',state_values)