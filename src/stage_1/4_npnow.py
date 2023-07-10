import numpy as np

w = np.random.random((2,3))
lr = 10e-3
t = np.array((2.0, 4.0))

for _ in range(1000):
    i = np.array((1.0, 2.0, 3.0))
    o = w.dot(i)
    e = t - o
    delta = lr * e * w.T
    w += delta.T

print(w)
print(o)
print(e)

