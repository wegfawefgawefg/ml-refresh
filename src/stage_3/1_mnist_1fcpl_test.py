import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import torch

from mnist.MNIST_UTILS import MNIST_UTILS

device = torch.device("cuda")

raw_inputs = MNIST_UTILS.getTestingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTestingLabels()

inputs = torch.tensor(inputs, device=device, dtype=float)
labels = torch.tensor(labels, device=device, dtype=int)


model_path = "./nnpt_mnist_1fcl.weights"

w1 = torch.load(model_path)

num_correct = 0
for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    l = labels[ii]

    o1 = torch.tensordot(i, w1, dims=1)
    ol = torch.argmax(o1)

    if ol == l:
        num_correct += 1

    # print(f"{l}: {ol}")

percent_correct = 100 * num_correct / len(inputs)
print(f"accuracy: {percent_correct}")
