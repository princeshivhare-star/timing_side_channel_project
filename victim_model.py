import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_layers=2, output_size=10):
        super().__init__()

        layers = []
        in_features = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_model(num_layers):
    return SimpleMLP(num_layers=num_layers)


def run_inference(model, batch_size, repeats=50):
    x = torch.randn(batch_size, 784)

    with torch.no_grad():
        for _ in range(repeats):
            y = model(x)