import numpy as np

lr = 1e-3

i = np.array((1.0, 2.0))

w1 = np.random.random((2, 3))
w2 = np.random.random((3, 4))
w3 = np.random.random((4, 2))

t = np.array((5.0, 6.0))


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

    w3d = np.outer(o2, w3g)
    w2d = np.outer(o1, w2g)
    w1d = np.outer(i, w1g)

    w3 += lr * w3d
    w2 += lr * w2d
    w1 += lr * w1d

print(w3g.sum())
