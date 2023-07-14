import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import math
import torch

model_path = "./nnpt_mnist_1fcl.weights"
w1 = torch.load(model_path).cpu().detach().numpy()  # 784,10
w1 = w1.T.reshape((28 * 10, 28))

print(w1.shape)
red = (w1 <= 0.0) * -w1 * 100.0
green = (w1 >= 0.0) * w1 * 100.0
blue = np.zeros(w1.shape)
wc = np.dstack((red, green, blue))

print(wc.shape)
wc = (wc * 64).astype(np.uint8)

image = Image.frombuffer("RGB", (28, 28 * 10), wc, "raw", "RGB", 0, 1)
image = image.resize((500, 500 * 10))
image.save("t_fcl1_wc.png")

# image = Image.fromarray(w1_scaled)
