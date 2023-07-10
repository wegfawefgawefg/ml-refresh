import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import math

from mnist.MNIST_UTILS import MNIST_UTILS


raw_inputs = MNIST_UTILS.getTrainingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

"""
# a = raw_inputs[0:10]
# l = labels[0:10]
# print(l)
# quit()
# print(a)

# a = a.reshape((28 * 10, 28))
# plt.imshow(a, cmap="gray", vmin=0, vmax=255)
# plt.axis("off")
# plt.savefig("grayscale_image.png")
"""

model_path = "./nn_mnist_1fcl.weights.npy"
w1 = np.load(model_path)  # 784,10
# w1 = w1.reshape((28 * 28, 10))
w1 = w1.T.reshape((28 * 10, 28))

# print(w1.max())

# w1_scaled = (w1 * 128).astype(np.uint8)
# image = Image.fromarray(w1_scaled)
# image.save("w1_image.png")


# w1 = w1 > 0.01

# plt.imshow(w1, cmap="gray", vmin=0, vmax=1)
# plt.axis("off")
# plt.savefig("model_weights.png")

# image = Image.open("model_weights.png")
# image.show()

print(w1.shape)
red = (w1 <= 0.0) * -w1 * 100.0
green = (w1 >= 0.0) * w1 * 100.0
blue = np.zeros(w1.shape)
wc = np.dstack((red, green, blue))

print(wc.shape)
wc = (wc * 64).astype(np.uint8)

image = Image.frombuffer("RGB", (28, 28 * 10), wc, "raw", "RGB", 0, 1)
image = image.resize((500, 500 * 10))
image.save("fcl1_wc.png")

# image = Image.fromarray(w1_scaled)
