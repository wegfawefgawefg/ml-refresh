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

inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
labels = torch.tensor(labels, device=device, dtype=torch.int32)

model_path = "./model.ptm"

# model = torch.nn.Sequential(
#     torch.nn.Linear(784, 128),
#     torch.nn.Linear(128, 10),
# ).to(device)
from model_4 import Network

model = Network(0.000, input_shape=784, output_shape=10)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

num_correct = 0
for ii in tqdm(range(len(inputs))):
    i = inputs[ii]
    l = labels[ii]

    pred = model(i)
    ol = torch.argmax(pred)

    if ol == l:
        num_correct += 1

    # print(f"{l}: {ol}")

percent_correct = 100 * num_correct / len(inputs)
print(f"accuracy: {percent_correct}")
