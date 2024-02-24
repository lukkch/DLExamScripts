import numpy as np
import scipy.special

queries = np.array([
    [1, 2, 2, 2],
    [0, 4, 0, 4]
])

keys = np.asarray([
    [2, 4, 4, 1],
    [1, 2, 3, 4]
])

values = np.array([
    [1, 1, 2, 2],
    [0, 3, 3, 8]
])

# embedding dimension
d_k = 4



assert(len(queries) == len(keys) == len(values))

before_softmax = (queries @ np.transpose(keys)) / np.sqrt(d_k)
z = scipy.special.softmax(before_softmax, axis=1) @ values

for i in range(len(z)):
    print(f"z_{i}: {z[i]}")
