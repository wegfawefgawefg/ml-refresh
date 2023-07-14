import torch
from torchvision import transforms
import os
from tqdm import tqdm
import subprocess
from torchvision.transforms import functional as F


duration_in_seconds = 60
fps = 60

num_frames = fps * duration_in_seconds

# Load the model
model_path = "./model.ptm"
from model_4 import Network

device = torch.device("cuda:0")
dshape = 784 * 3
model = Network(0.0001, input_shape=dshape, output_shape=dshape).to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Make a container for the output frames
frames = torch.zeros((num_frames, dshape), dtype=torch.float32, device=device)
starter_frame = torch.rand((dshape), dtype=torch.float32, device=device)
frames[0] = starter_frame

scale_factor = 1.80  # How much to zoom on each frame. Adjust as needed.
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((int(28 * scale_factor), int(28 * scale_factor))),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
    ]
)

# generate
bar = tqdm(range(frames.shape[0] - 1))
for ii in bar:
    i = frames[ii]

    #########
    # variation needs to be added between frames to increase entropy
    #########

    # 1 PIXEL ZOOM METHOD
    i = i.view(3, 28, 28).cpu()
    i = transform(i).to(device).flatten()

    # NOISE METHOD
    i = i + torch.rand((dshape), dtype=torch.float32, device=device) * 0.5

    o = model(i)
    frames[ii + 1] = o

# render video
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

frames = frames.cpu()
bar = tqdm(range(frames.shape[0]))
for ii in bar:
    frame = frames[ii]
    frame = frame.view(3, 28, 28)
    frame_image = transforms.ToPILImage()(frame)

    # Save the PIL image as a PNG file
    file_path = os.path.join(output_folder, f"frame_{ii + 1:05d}.png")
    frame_image.save(file_path)

# Use FFmpeg to concatenate the frames into a video
output_video = "output_video.mp4"
ffmpeg_cmd = f"ffmpeg -r {fps} -i {output_folder}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {output_video} -y"
subprocess.call(ffmpeg_cmd, shell=True)
