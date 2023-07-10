import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from mnist.MNIST_UTILS import MNIST_UTILS

raw_inputs = MNIST_UTILS.getTrainingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

lr = 0.01  # lr = 0.005


def relu(x):
    return x * (x > 0)


def derelu(y):
    return y > 0


model_path = "./nn_mnist_2fcl.weights.npz"
HL_SIZE = 256

if LOAD := False:
    model = np.load(model_path)
    w1 = model["w1"]
    w2 = model["w2"]
else:
    w1 = 0.001 * (2.0 * np.random.random((784, HL_SIZE)) - 1.0)
    w2 = 0.001 * (2.0 * np.random.random((HL_SIZE, 10)) - 1.0)

for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    l = labels[ii]
    t = targets[ii]

    o1 = relu(i.dot(w1))
    o2 = o1.dot(w2)

    w2g = t - o2
    w1g = w2g.dot(w2.T) * derelu(o1)

    w2d = np.outer(o1, w2g)
    w1d = np.outer(i, w1g)

    w2 += lr * w2d
    w1 += lr * w1d

    if ii % 1000 == 0:
        se = np.sum(np.square(w2g))
        print(f"e: {se}")

if SAVE := True:
    np.savez(model_path, w1=w1, w2=w2)
