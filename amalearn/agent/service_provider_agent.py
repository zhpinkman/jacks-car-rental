import numpy as np
from amalearn.agent import AgentBase
from scipy.stats import poisson
import copy

class ServiceProviderAgent(AgentBase):
    def __init__(self, id, environment):
        super(ServiceProviderAgent, self).__init__(id, environment)
        self.max_capacity = 20
        self.max_transfer = 5
        self.state_values = np.zeros((self.max_capacity + 1, self.max_capacity + 1))
        self.policy = np.zeros((self.max_capacity + 1, self.max_capacity + 1), dtype=int)
        self.return_poisson_lambda_first_loc = 3
        self.return_poisson_lambda_second_loc = 2
        self.request_poisson_lambda_first_loc = 3
        self.request_poisson_lambda_second_loc = 4
        self.discount_factor = .9
        self.theta = 1e-1
        self.reward_per_customer = 10
        self.punish_per_transfer = 2
        self.actions = np.arange(- self.max_transfer, self.max_transfer + 1)
        self.poisson_upper_bound = 10
        self.poisson_cache = dict()

    def poisson_probability(self, n, lam):
        key = n * 10 + lam
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]


    def return_calculator(self, state, action):

        net_return = 0.0
        net_return -= np.abs(action) * self.punish_per_transfer
        
        first_loc_capacity = state[0]
        second_loc_capacity = state[1]

        first_loc_capacity = min(first_loc_capacity - action, self.max_capacity)
        second_loc_capacity = min(second_loc_capacity + action, self.max_capacity)

        for request_outcome1 in range(self.poisson_upper_bound):
            for request_outcome2 in range(self.poisson_upper_bound):
                request_prob = self.poisson_probability(request_outcome1, self.request_poisson_lambda_first_loc) * \
                    self.poisson_probability(request_outcome2, self.request_poisson_lambda_second_loc)

                accepted_request_first_loc = min(request_outcome1, first_loc_capacity)
                accepted_request_second_loc = min(request_outcome2, second_loc_capacity)

                returned_reward = (accepted_request_first_loc + accepted_request_second_loc) * self.reward_per_customer

                remained_capacity_first_loc = first_loc_capacity - accepted_request_first_loc
                remained_capacity_second_loc = second_loc_capacity - accepted_request_second_loc

                for return_outcome1 in range(self.poisson_upper_bound):
                    for return_outcome2 in range(self.poisson_upper_bound):
                        return_prob = self.poisson_probability(return_outcome1, self.return_poisson_lambda_first_loc) * \
                            self.poisson_probability(return_outcome2, self.return_poisson_lambda_second_loc)

                        net_return += request_prob * return_prob * (returned_reward + self.discount_factor * \
                            self.state_values[
                                    min(remained_capacity_first_loc + return_outcome1, self.max_capacity), 
                                    min(remained_capacity_second_loc + return_outcome2, self.max_capacity)
                                ])
        return net_return



    
    def policy_evaluation(self):
        while True:
            old_value = copy.deepcopy(self.state_values)
            for i in range(self.max_capacity + 1):
                for j in range(self.max_capacity + 1):
                    new_state_value = self.return_calculator([i, j], self.policy[i, j])
                    self.state_values[i, j] = new_state_value
            max_value_change = abs(old_value - self.state_values).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < self.theta:
                break
        return



        

    def policy_improvement(self):
        self.policy_evaluation()
        policy_stable = True
        for i in range(self.max_capacity + 1):
            for j in range(self.max_capacity + 1):
                old_action = copy.deepcopy(self.policy[i, j])
                action_returns = []
                for action in self.actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(self.return_calculator(
                            (i, j), action
                        ))
                    else:
                        action_returns.append(- np.inf)
                new_action = self.actions[np.argmax(action_returns)]
                self.policy[i, j] = new_action
                if self.policy[i, j] != old_action:
                    policy_stable = False
        return policy_stable


    def sweep(self):
        policy_stable = self.policy_improvement()
        return self.state_values, self.policy, policy_stable



    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()
        action = np.random.choice(available_actions)
        obs, r, d, i = self.environment.step(action)
        print(obs, r, d, i)
        self.environment.render()
        return obs, r, d, i