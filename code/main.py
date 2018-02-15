import numpy as np
from supply_distribution import SupplyDistribution

model = SupplyDistribution()
for action in model.action_space():
    print(action)
print(type(model.action_space()))
action1 = np.array(model.action_space()[0])
print(type(action1))
print("current state 0 {}".format(model.s))
print("action1 {}".format(action1))
print("demand {}".format(model.demand))
model.step(action1)
print("current state 1 {}".format(model.s))
action2 = np.array(model.action_space()[10])
print("action2 {}".format(action2))
print("demand {}".format(model.demand))
model.step(action2)
print("current state 2 {}".format(model.s))
