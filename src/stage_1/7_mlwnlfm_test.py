import numpy as np

lr = 1e-3

i = np.array((1.0, 2.0))

w1 = np.random.random((2, 3))
w2 = np.random.random((3, 2))
w3 = np.random.random((2, 1))

t = np.array((5.0))


def relu(x):
    return (x > 0) * x


def derelu(x):
    return x > 0


for _ in range(1000):
    o1 = relu(i.dot(w1))
    o2 = relu(o1.dot(w2))
    o3 = o2.dot(w3)

    w3g = t - o3
    w2g = w3g.dot(w3.T) * derelu(o2)
    w1g = w2g.dot(w2.T) * derelu(o1)

    w3 += lr * np.outer(o2.T, w3g)
    w2 += lr * np.outer(o1.T, w2g)
    w1 += lr * np.outer(i.T, w1g)

print(w3g)
