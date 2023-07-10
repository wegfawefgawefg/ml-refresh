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


models = []
for m in range(8):
    model_path = "./nn_mnist_1fcl" + "_" + str(m) + ".weights.npy"
    models.append(np.load(model_path))

num_correct = 0
for ii in tqdm(range(len(inputs))):
    # for ii in range(5):
    i = inputs[ii]
    l = labels[ii]

    outputs = []
    for m in range(8):
        w = models[m]
        o1 = i.dot(w)
        outputs.append(o1)

    assert not np.array_equal(outputs[0], outputs[1])
    vote = sum(outputs)
    ol = np.argmax(vote)

    if ol == l:
        num_correct += 1

    # print(f"{l}: {ol}")

percent_correct = 100 * num_correct / len(inputs)
print(f"accuracy: {percent_correct}")
