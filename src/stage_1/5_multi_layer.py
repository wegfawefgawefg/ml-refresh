import numpy as np

"""mmm pipe alignment"""
lr = 10e-3

w1 = np.random.random((3, 2))
w2 = np.random.random((3, 3))
w3 = np.random.random((1, 3))

i = np.array((1.0, 2.0))
t = np.array((1.0))

for _ in range(1000):
    print(f"w1-i: {w1.shape}-{i.shape}")
    o1 = i.dot(w1.T)
    print(f"o1: {o1.shape}")
    print(f"w2-o1: {w2.shape}-{o1.shape}")
    o2 = o1.dot(w2.T)
    print(f"o2: {o2.shape}")
    print(f"w3-o2: {w3.shape}-{o2.shape}")
    o3 = o2.dot(w3.T)
    print(f"o3: {o3.shape}")

    e = t - o3
    print(e)

    # Gradients for the weights
    l3l = t - o3  # derivative of loss wrt o3
    l2l = l3l.dot(w3)  # derivative of loss wrt o2
    l1l = l2l.dot(w2)  # derivative of loss wrt o1

    l3d = np.outer(l3l, o2)  # derivative of loss wrt w3
    l2d = np.outer(l2l, o1)  # derivative of loss wrt w2
    l1d = np.outer(l1l, i)  # derivative of loss wrt w1

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
