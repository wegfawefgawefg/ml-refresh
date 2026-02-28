import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(torch.nn.Module):
    DROPOUT_FRAC = 0.1

    # HIDDEN_SIZE = 128
    # HIDDEN_SIZE = 256
    # HIDDEN_SIZE = 512
    HIDDEN_SIZE = 1024

    def __init__(self, alpha, input_shape, output_shape):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = nn.Sequential(
            nn.Linear(input_shape, Network.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(Network.DROPOUT_FRAC),
            nn.Linear(Network.HIDDEN_SIZE, Network.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(Network.DROPOUT_FRAC),
            nn.Linear(Network.HIDDEN_SIZE, Network.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(Network.DROPOUT_FRAC),
            nn.Linear(Network.HIDDEN_SIZE, output_shape),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss(reduction="sum")
        self = self.to(self.device)
        for param in self.parameters():
            param.to(self.device)

    def forward(self, x):
        y = self.layers(x)
        return y


if __name__ == "__main__":
    input_shape = (1, 784)
    net = Network(alpha=0.001, input_shape=784, output_shape=784 * 3)

    x = torch.ones(input_shape).to(net.device)
    print(x.shape)

    y = net(x)
    print(y)
    print(f"y.shape: {y.shape}")

    print(net.device)
