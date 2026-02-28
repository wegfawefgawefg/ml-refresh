import os
import subprocess

import torch
from model import Network
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

duration_in_seconds = 16
fps = 60
num_frames = fps * duration_in_seconds


frame_dim = 128
model_path = f"./model_{frame_dim}.ptm"
device = torch.device("cuda:0")

dshape = frame_dim * frame_dim * 3

model = Network(0.0001, input_shape=dshape, output_shape=dshape).to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

scale_factor = 1.80  # How much to zoom on each frame. Adjust as needed.
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((int(28 * scale_factor), int(28 * scale_factor))),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
    ]
)

################################################################################
# GENERATE FRAMES
################################################################################
output_folder = "generated_frames"
if os.path.exists(output_folder):
    subprocess.call(f"rm -rf {output_folder}", shell=True)
os.makedirs(output_folder, exist_ok=True)

frame = torch.rand((dshape), dtype=torch.float32, device=device).detach()

bar = tqdm(range(num_frames))
for ii in bar:
    #########
    # variation needs to be added between frames to increase entropy
    #########

    # 1 PIXEL ZOOM METHOD
    # i = i.view(3, 28, 28).cpu()
    # i = transform(i).to(device).flatten()

    # NOISE METHOD
    n = 0.3
    frame = (
        frame * (1.0 - n) + torch.rand((dshape), dtype=torch.float32, device=device) * n
    )
    # clamp high values
    frame = torch.clamp(frame, 0.0, 1.0)

    frame = model(frame)

    # save the frame
    cf = frame.cpu()
    cf = cf.view(3, frame_dim, frame_dim)
    frame_image = transforms.ToPILImage()(cf)

    # Save the PIL image as a PNG file
    file_path = os.path.join(output_folder, f"{ii + 1:05d}.png")
    frame_image.save(file_path)

# Use FFmpeg to concatenate the frames into a video
output_video = "output_video.mp4"
ffmpeg_cmd = f"ffmpeg -r {fps} -i {output_folder}/%05d.png -c:v libx264 -pix_fmt yuv420p {output_video} -y"
subprocess.call(ffmpeg_cmd, shell=True)
