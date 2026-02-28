import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda:0")


transform = transforms.Compose([transforms.ToTensor()])

#     image_list = []
#     for file_name in tqdm(files):
#         file_path = os.path.join(path, file_name)
#         image = Image.open(file_path)
#         image = transform(image)
#         image_list.append(image)
#     frames = torch.stack(image_list)
#     frames = frames.view(frames.size(0), -1)


# print(frames.shape)
# frames = frames.to(device)

lr = 0.00001
# lr = 0.001

from model import Network

frame_dim = 128
model_path = f"./model_{frame_dim}.ptm"
dshape = frame_dim * frame_dim * 3
model = Network(lr, input_shape=dshape, output_shape=dshape).to(device)

# get num frames in the frame path
path = f"./videos/secret_life_of_pets/{frame_dim}_{frame_dim}/frames"
num_frames = len([file for file in os.listdir(path) if file.endswith(".png")])


class SelfCleaningCache:
    """
    A cache that will remove items that havent been access in n queries
    """

    def __init__(
        self,
        miss_function,
        age_threshold=2,
    ):
        self.age_threshold = age_threshold
        self.cache = {}
        self.ages = {}
        self.miss_function = miss_function

    def get(self, key):
        # increment all ages
        for k in self.ages:
            self.ages[k] += 1

        for k in list(self.ages.keys()):
            if self.ages[k] > self.age_threshold:
                del self.ages[k]
                del self.cache[k]

        if key in self.cache:
            self.ages[key] = 0
            return self.cache[key]
        else:
            self.cache[key] = self.miss_function(key)
            self.ages[key] = 0
            return self.cache[key]


def load_frame(i):
    file_path = os.path.join(path, f"{i}.png")
    image = Image.open(file_path)
    image = transform(image)
    image = image.view(-1)
    image = image.to(device)
    return image


cache = SelfCleaningCache(load_frame, age_threshold=4)

num_frames = num_frames - 1
# randomize the order of the frames, dont use numpy or pytorch
frames = list(range(1, num_frames))
np.random.shuffle(frames)
# convert to list
num_s = int(10_000 * (1.0 + model.DROPOUT_FRAC))
frames = list(frames)[0:num_s]

losses = []
# bar = tqdm(range(1, num_frames - 1))
bar = tqdm(frames)
for ii in bar:
    i = cache.get(ii)
    t = cache.get(ii + 1)

    pred = model(i)
    loss = model.loss(pred, t)
    losses.append(loss.item())
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


"""
notes:
- the loss is a little lower than expected, even after few iterations, 
    this could be because 

"""
