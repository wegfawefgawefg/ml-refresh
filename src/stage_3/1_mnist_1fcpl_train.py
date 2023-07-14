import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import torch

from mnist.MNIST_UTILS import MNIST_UTILS

device = torch.device("cuda")

raw_inputs = MNIST_UTILS.getTrainingData()
inputs = raw_inputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

inputs = torch.tensor(inputs, device=device, dtype=float)
targets = torch.tensor(targets, device=device, dtype=float)

lr = 0.0001

model_path = "./nnpt_mnist_1fcl.weights"

w1 = 0.001 * torch.randn((784, 10), device=device, dtype=float)

for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    t = targets[ii]

    o1 = torch.tensordot(i, w1, dims=1)

    w1g = t - o1

    w1d = torch.ger(i, w1g)

    w1 += lr * w1d

if SAVE := True:
    torch.save(w1, model_path)
