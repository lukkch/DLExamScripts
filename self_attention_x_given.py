import numpy as np
import scipy.special

w_q = np.array([[0.0],[2.0]])
w_k = np.array([[2.0],[0.0]])
w_v = np.array([[0.5],[0.5]])
w_o = np.array([[0.5],[0.5]])

# embedding dimension
d_k = 1

x = np.array([[1, 1], [0, 0], [1, 1]])

q = x @ w_q
k = x @ w_k
v = x @ w_v

before_softmax = (q @ k.T) / np.sqrt(d_k)
z = scipy.special.softmax(before_softmax, axis=1) @ v
out = z @ w_o.T

for i in range(len(out)):
    print(f"z_{i}: {out[i]}")

