import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import torch
import os
from PIL import Image
from torchvision import transforms
import pickle
from tqdm import tqdm

device = torch.device("cuda:0")

"""
- probably data should be randomized pairs?
    make a pair index generator
    reduce data ordering dependency (industry standard but philosophically impure?)
- need output viewer
- mini batches might speed up, reduce noise
"""

##################################################################
#  DATA WRANGLE
##################################################################

path = "/home/vega/Coding/Training/ml-refresh/src/sq_nextframe_1/videos/segretlife"
data_checkpoint = "frametensor"

LOAD = True
if not LOAD:
    # load every frame
    files = [file for file in os.listdir(path) if file.endswith(".png")]
    files.sort()

    transform = transforms.Compose(
        [
            # transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]
    )

    image_list = []
    for file_name in tqdm(files):
        file_path = os.path.join(path, file_name)
        image = Image.open(file_path)
        image = transform(image)
        image_list.append(image)
    frames = torch.stack(image_list)
    frames = frames.view(frames.size(0), -1)

    with open(data_checkpoint, "wb") as pf:
        pickle.dump(frames, pf)
else:
    with open(data_checkpoint, "rb") as file:
        frames = pickle.load(file)

print(frames.shape)
frames = frames.to(device)

##################################################################
#  TRAIN
##################################################################
lr = 0.0001
model_path = "./model.ptm"

from model_4 import Network

dshape = 784 * 3
model = Network(lr, input_shape=dshape, output_shape=dshape).to(device)
frames = frames.to(model.device)

# for i, layer in enumerate(model.children()):
#     model[i].weight.data * 0.001

losses = []
bar = tqdm(range(frames.shape[0] - 1))
for ii in bar:
    i = frames[ii]
    t = frames[ii + 1]

    pred = model(i)
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
