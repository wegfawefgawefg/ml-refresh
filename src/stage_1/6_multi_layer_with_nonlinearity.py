import numpy as np

"""add a nonlinearity"""


def relu(x):
    return (x > 0) * x


def derRelu(x):
    return x > 0


lr = 1e-3

w1 = np.random.random((2, 3))
w2 = np.random.random((3, 2))
w3 = np.random.random((2, 1))

i = np.array((1.0, 2.0))
t = np.array((1.0))

for _ in range(1):
    print(f"w1-i: {w1.shape}-{i.shape}")
    o1 = relu(i.dot(w1))
    print(f"o1: {o1.shape}")
    print(f"w2-o1: {w2.shape}-{o1.shape}")
    o2 = relu(o1.dot(w2))
    print(f"o2: {o2.shape}")
    print(f"w3-o2: {w3.shape}-{o2.shape}")
    o3 = o2.dot(w3)
    print(f"o3: {o3.shape}")

    e = t - o3
    print(e)

    l3g = t - o3
    l2g = l3g.dot(w3.T) * derRelu(o2)
    l1g = l2g.dot(w2.T) * derRelu(o1)

    l3d = np.outer(o2.T, l3g)
    l2d = np.outer(o1.T, l2g)
    l1d = np.outer(i.T, l1g)

    # Update weights
    w3 += lr * l3d
    w2 += lr * l2d
    w1 += lr * l1d

print(i)
print(w1)
print(w2)
print(w3)
print(o3)
print(t)
