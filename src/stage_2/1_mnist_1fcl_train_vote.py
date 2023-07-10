import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from mnist.MNIST_UTILS import MNIST_UTILS

raw_inputs = MNIST_UTILS.getTrainingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

lr = 1e-4


def relu(x):
    return x * (x > 0)


def derelu(x):
    return x > 0


for m in range(8):
    model_path = "./nn_mnist_1fcl" + "_" + str(m) + ".weights"
    w1 = 0.001 * (2.0 * np.random.random((784, 10)) - 1.0)

    for ii in tqdm(range(len(inputs))):
        i = inputs[ii]
        l = labels[ii]
        t = targets[ii]

        o1 = i.dot(w1)

        w1g = t - o1

        w1d = np.outer(i, w1g)

        w1 += lr * w1d

    if SAVE := True:
        np.save(model_path, w1)
