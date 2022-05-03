import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 784
        self.encoder = nn.Sequential(
            nn.Linear(30, 16),  # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2))

        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 30))

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder

# Autoencoder setup
model_auto = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_auto.parameters(), lr=0.001)
num_epochs = 200

hidden = torch.tensor([[[-0.4707,  0.6851,  0.1866, -0.2789, -0.4853, -0.4037,  0.0495,
                   0.5666,  0.7394, -0.7765,  0.6090,  0.1975, -0.3077, -0.5115,
                   0.5862,  0.1491,  0.1997,  0.3768,  0.6755, -0.2807,  0.1284,
                   0.2788,  0.6721, -0.6597, -0.6864, -0.1767, -0.7467, -0.4301,
                   0.4984,  0.2274]]], device='cuda:0')

# train autoencoder
for epoch in range(num_epochs):
    recon = model_auto(hidden)
    loss = criterion(recon, hidden)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print(f'Epoch:{epoch + 1}, loss:{loss.item():.4f}')

print(hidden)
print(recon)