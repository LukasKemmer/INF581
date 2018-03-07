import numpy as np
import time
from supply_distribution import SupplyDistribution

model = SupplyDistribution()
#for action in model.action_space():
#    print(action)
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


model2 = SupplyDistribution()
action3 = np.array([0, 5, 0, 0])
model2.step(action3)
print("current state model2 {}".format(model2.s))
print("actions: ")
#for action in model2.action_space():
#    print(action)

model3 = SupplyDistribution()
t0_itter = time.time()
action_itter = model3.action_space_itertools()
t1_itter = time.time()

t0_recur = time.time()
action_recur = model3.action_space_recur()
t1_recur = time.time()
print("itter actions size: {}".format(action_itter.size))
print("itter actions time: {}".format(t1_itter - t0_itter))

print("recur actions size: {}".format(action_recur.size))
print("recur actions time: {}".format(t1_recur - t0_recur))

for action in action_recur:
    print(action)