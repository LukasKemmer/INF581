import numpy as np


class SupplyDistribution:
    """
    The supply distribution environment
    """

    def __init__(self, n_stores=5, cap_truck=100, prod_cost=1, store_cost = np.array([0.01, 0.1, 0.1, 0.1, 0.1, 0.1]),
                 truck_cost = np.array([2, 3, 4, 5, 6]), cap_store = np.array([500, 25, 25, 25, 25, 25]),
                 penalty_cost = 0.3, price = 2, discounting_factor = 0.5):
        """

        :param n_stores:
        :param cap_truck:
        :param prod_cost:
        :param store_cost:
        :param truck_cost:
        :param cap_store:
        :param penalty_cost:
        :param price:
        :param discounting_factor:
        """
        self.n_stores = n_stores
        self.s = np.zeros(self.n_stores + 1)
        self.demand = np.zeros(self.n_stores)
        # capacity
        self.cap_store = np.ones(n_stores + 1)
        self.cap_store = cap_store
        self.cap_truck = cap_truck
        # costs:
        self.prod_cost = prod_cost
        self.store_cost = np.array(store_cost)
        self.truck_cost = np.array(truck_cost)
        self.penalty_cost = penalty_cost
        self.price = price
        #other variables
        self.t = 0
        self.discounting_factor = discounting_factor
        self.reset()

    def reset(self):
        """
        Resets the environment to the starting conditions
        """
        self.s = np.zeros(self.n_stores + 1) # +1 Because the central warehouse is not counted as a stores but has a supply
        self.s[0] = self.cap_store[0] / 2  # start with center half full
        self.demand = np.zeros(self.n_stores)
        self.t = 0

    def step(self, action): # TODO Check np.array * -- Droche 15/02
        self.s[0] = np.min(self.s[0] + action[0] - sum(action[1:]), self.cap_store[0])
        self.s[1:] = np.minimum(self.s[1:] - self.demand + action[1:], self.cap_store[1:])
        reward = (sum(self.demand) * self.price
                  - action[0] * self.prod_cost
                  - np.sum(np.maximum(np.zeros(len(self.s)), self.s) * self.store_cost)
                  - np.sum(np.minimum(np.zeros(len(self.s)), self.s)) * self.penalty_cost
                  - np.sum(np.ceil(action / self.cap_truck).T * self.truck_cost))
        self.t += 1
        self.update_demand()
        done = 0
        info = "hello!" # TODO delete or do something -- Droche 15/02
        return self.s, reward, done, info

    def update_demand(self):
        """
        Updates the demand using the update demand function
        :return:
        """
        # TODO makes this function a parameter of the env so we can change it easy. Not necessary. --Droche 15/02/2018
        demand = np.zeros(self.n_stores)
        for i in range(self.n_stores):
            demand[i] = 30 * np.sin(2 * np.pi * self.t / 365 + i) + 30
        self.demand = demand

    def action_space(self):
        """
        :return: Returns the set of possibles actions that the agent can make
        """
        action = []
        return

    def observation_space(self):
        """
        Used to observe the current state of the environment
        :return:
        """
        return







