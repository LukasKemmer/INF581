import numpy as np
import itertools


class SupplyDistribution:
    """
    The supply distribution environment
    """

    def __init__(self, n_stores=3, cap_truck=100, prod_cost=1, max_prod=10,
                 store_cost=np.array([0.01, 0.1, 0.1, 0.1]), truck_cost=np.array([2, 3, 4]),
                 cap_store=np.array([20, 5, 5, 5]), penalty_cost=0.3, price=5, gamma=0.95):
        """
        :param n_stores:
        :param cap_truck:
        :param prod_cost:
        :param store_cost:
        :param truck_cost:
        :param cap_store:
        :param penalty_cost:
        :param price:
        """
        self.n_stores = n_stores
        self.s = np.zeros(self.n_stores + 1, dtype=int)
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.price = price
        self.max_prod = max_prod
        # capacity
        self.cap_store = np.ones(n_stores + 1, dtype=int)
        self.cap_store = cap_store
        self.cap_truck = cap_truck
        # costs:
        self.prod_cost = prod_cost
        self.store_cost = np.array(store_cost)
        self.truck_cost = np.array(truck_cost)
        self.penalty_cost = penalty_cost
        # other variables
        self.gamma = gamma
        self.t = 0

        self.reset()

    def reset(self):
        """
        Resets the environment to the starting conditions
        """
        self.s = np.zeros(self.n_stores + 1, dtype=int)  # +1 Because the central warehouse is not counted as a store
        # self.s[0] = self.cap_store[0] / 2  # start with center half full TODO decide initial values --Droche 15/02
        self.s[0] = 5
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.t = 0
        return self.s.copy()

    def step(self, action):  # TODO Check np.array * -- Droche 15/02
        self.s[0] = min(self.s[0] + action[0] - sum(action[1:]), self.cap_store[0])
        self.s[1:] = np.minimum(self.s[1:] - self.demand + action[1:], self.cap_store[1:])
        reward = (sum(self.demand) * self.price
                  - action[0] * self.prod_cost
                  - np.sum(np.maximum(np.zeros(len(self.s)), self.s) * self.store_cost)
                  - np.sum(np.minimum(np.zeros(len(self.s)), self.s)) * self.penalty_cost
                  - np.sum(np.ceil(action[1:] / self.cap_truck) * self.truck_cost)) # Removed .T after np.ceil, as it was unnecessary -- Luke 19/02
        info = "Demand was: ", self.demand  # TODO delete or do something -- Droche 15/02
        self.t += 1
        self.update_demand()
        done = 0
        return self.s.copy(), reward, done, info

    def update_demand(self):
        """
        Updates the demand using the update demand function
        :return:
        """
        # TODO makes this function a parameter of the env so we can change it easy. Not necessary. -- Droche 15/02/2018
        demand = np.zeros(self.n_stores, dtype=int)
        for i in range(self.n_stores):
            # We need an integer so we use the ceiling because if there is demand then we asume the users will buy
            # what they need and keep the rests. We use around to get an integer out of it.
            demand[i] = int(np.ceil(1.5 * np.sin(2 * np.pi * self.t / 52 + i) + 1.5 + np.random.randint(0, 2)))
        self.demand = demand

    def action_space(self):
        """
        Returns the set of possibles actions that the agent can make
        :return: The posible actions in a list of tuples. Each tuple with (a0, a1, ..., ak) k = n_stores.
        """
        action_space = []
        actions_list = [range(self.max_prod)]
        for i in range(self.n_stores):
            actions_list.append(range(self.s[0]))

        for element in itertools.product(*actions_list):
            if sum(element[1:]) <= self.s[0]:
                action_space.append(element)
        return action_space

    def observation_space(self):
        """
        Used to observe the current state of the environment
        :return:
        """
        return

