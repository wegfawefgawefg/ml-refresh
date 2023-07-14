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
model_path = "./model.ptm"

from model_4 import Network


dshape = 784 * 3
model = Network(lr, input_shape=dshape, output_shape=dshape).to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

frames = frames.to(model.device)

accuracies = []

bar = tqdm(range((frames.shape[0] - 1) // 8))
for ii in bar:
    i = frames[ii]
    t = frames[ii + 1]

    pred = model(i)

    mse = torch.mean((pred - t) ** 2)
    accuracies.append(mse)

    # avg_accuracy = (avg_accuracy * ii + mse.item()) / (ii + 1)

    bar.set_description(f"avg accuracy: {mse:.4f}")

avg_accuracy = sum(accuracies) / len(accuracies)
print(f"Final average accuracy: {avg_accuracy:.4f}")
