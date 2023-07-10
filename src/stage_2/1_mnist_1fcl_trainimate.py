import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

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


model_path = "./nn_mnist_1fcl.weights"

if LOAD := False:
    w1 = np.load(model_path)
else:
    w1 = 0.001 * (2.0 * np.random.random((784, 10)) - 1.0)

frames_path = "./frames"
for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    l = labels[ii]
    t = targets[ii]

    o1 = i.dot(w1)

    w1g = t - o1

    w1d = np.outer(i, w1g)

    w1 += lr * w1d

    # generate a frame
    wi = w1.T.reshape((28 * 10, 28))
    red = (wi <= 0.0) * -wi * 100.0
    green = (wi >= 0.0) * wi * 100.0
    blue = np.zeros(wi.shape)
    wc = np.dstack((red, green, blue))
    wc = (wc * 64).astype(np.uint8)
    image = Image.frombuffer("RGB", (28, 28 * 10), wc, "raw", "RGB", 0, 1)
    # image = image.resize((500, 500 * 10))
    imname = f"{ii}.png"
    impath = os.path.join(frames_path, imname)
    image.save(impath)

# image = Image.fromarray(w1_scaled)
