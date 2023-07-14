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

inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
targets = torch.tensor(targets, device=device, dtype=torch.float32)

lr = 0.01

model_path = "./nnpt_2_mnist_1fcpl.weights"

w1 = 0.001 * torch.randn((784, 10), device=device, dtype=torch.float32)
w1.requires_grad = True


losses = []
bar = tqdm(range(len(inputs)))
for ii in bar:
    i = inputs[ii]
    t = targets[ii]

    o1 = torch.matmul(i, w1)

    loss = torch.mean((t - o1) ** 2)
    losses.append(loss)
    if len(losses) > 10:
        losses.pop(0)

    if ii % 100 == 0:
        avg_loss = sum(losses) / len(losses)
        bar.set_description(f"avg loss: {avg_loss:.4f}")

    loss.backward()
    with torch.no_grad():
        w1 -= lr * w1.grad
    w1.grad.zero_()

if SAVE := True:
    torch.save(w1, model_path)
