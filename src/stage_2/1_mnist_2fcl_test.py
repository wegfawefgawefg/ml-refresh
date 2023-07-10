import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from mnist.MNIST_UTILS import MNIST_UTILS

raw_inputs = MNIST_UTILS.getTestingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTestingLabels()


def relu(x):
    return x * (x > 0)


model_path = "./nn_mnist_2fcl.weights.npz"

model = np.load(model_path)
w1 = model["w1"]
w2 = model["w2"]

num_correct = 0
for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    l = labels[ii]

    o1 = relu(i.dot(w1))
    o2 = o1.dot(w2)

    ol = np.argmax(o2)

    if ol == l:
        num_correct += 1

    # print(f"{l}: {ol}")

percent_correct = 100 * num_correct / len(inputs)
print(f"accuracy: {percent_correct}")
