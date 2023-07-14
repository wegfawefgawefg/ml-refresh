import numpy as np
from PIL import Image
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(784, 10),
)

model_path = "./model.ptm"
model.load_state_dict(torch.load(model_path))
model.eval()

model = model.cpu()

# Obtain weights from the first layer of the model
w1 = model[0].weight.data.cpu().numpy()  # 10,784
w1 = w1.T
w1 = w1.T.reshape((28 * 10, 28))

red = (w1 <= 0.0) * -w1 * 100.0
green = (w1 >= 0.0) * w1 * 100.0
blue = np.zeros(w1.shape)
wc = np.dstack((red, green, blue))

wc = (wc * 64).astype(np.uint8)

image = Image.frombuffer("RGB", (28, 28 * 10), wc, "raw", "RGB", 0, 1)
image = image.resize((500, 500 * 10))
image.save("model_weights.png")
