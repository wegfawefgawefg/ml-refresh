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

lr = 0.001
model_path = "./model.ptm"

# model = torch.nn.Sequential(
#     torch.nn.Linear(784, 128),
#     torch.nn.Linear(128, 10),
# ).to(device)
from model_4 import Network

model = Network(lr, input_shape=784, output_shape=10)
# for i, layer in enumerate(model.children()):
#     model[i].weight.data * 0.001

# loss_fn = torch.nn.MSELoss(reduction="sum")
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

losses = []
bar = tqdm(range(len(inputs)))
for ii in bar:
    i = inputs[ii]
    t = targets[ii]

    pred = model(i)
    # loss = loss_fn(pred, t)
    loss = model.loss(pred, t)
    losses.append(loss)
    if len(losses) > 10:
        losses.pop(0)

    if ii % 100 == 0:
        avg_loss = sum(losses) / len(losses)
        bar.set_description(f"avg loss: {avg_loss:.4f}")

    # model.zero_grad()
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= lr * param.grad

if SAVE := True:
    torch.save(model.state_dict(), model_path)
