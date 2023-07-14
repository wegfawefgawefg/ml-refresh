import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(torch.nn.Module):
    HIDDEN_SIZE = 128

    def __init__(self, alpha, input_shape, output_shape):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, Network.HIDDEN_SIZE)
        self.fc2 = nn.Linear(Network.HIDDEN_SIZE, output_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input_shape = (1, 784)
    net = Network(alpha=0.001, input_shape=784, output_shape=10)

    x = torch.ones(input_shape).to(net.device)
    print(x.shape)

    y = net(x)
    print(y)
    print(f"y.shape: {y.shape}")
