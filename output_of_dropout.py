import numpy as np


w = np.array([
    [1, 0.4], # first unit
    [0, 0]    # second unit -> drop if dropout mask = [1, 0]
])
x = np.array([-1, 2])

p_delete = 0.5

scaling = 1/(1-p_delete)

print((w @ x) * scaling)