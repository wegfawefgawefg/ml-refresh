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
import json
import shutil

device = torch.device("cuda:0")

##################################################################
#  DATA WRANGLE
##################################################################

path = "./videos/secret_life_of_pets/128_128/frames"
output_folder = "./frame_tensors"
data_checkpoint = output_folder + "/frame_tensor_batch_"
batch_size = 4096
metadata = {"batch_size": batch_size, "num_batches": 0}

# clear output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# load every frame
files = [file for file in os.listdir(path) if file.endswith(".png")]
files.sort()

transform = transforms.Compose([transforms.ToTensor()])

image_list = []
for idx, file_name in enumerate(tqdm(files)):
    file_path = os.path.join(path, file_name)
    image = Image.open(file_path)
    image = transform(image)
    image_list.append(image)

    # Save and clear list every batch_size files
    if (idx + 1) % batch_size == 0 or idx + 1 == len(files):
        batch = torch.stack(image_list)
        batch = batch.view(batch.size(0), -1)
        with open(f"{data_checkpoint}{metadata['num_batches']}.pickle", "wb") as pf:
            pickle.dump(batch, pf)

        image_list = []
        metadata["num_batches"] += 1

# Save metadata to a json file
with open(os.path.join(".", "metadata.json"), "w") as f:
    json.dump(metadata, f)
