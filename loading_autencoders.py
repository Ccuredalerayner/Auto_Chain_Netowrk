import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 784
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3))  # N, 3

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid())  # sigmoid as values are between 0-1

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder


class ConcatAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder


fileA = 'saves/Autoencoder_A.pth'
fileB = 'saves/Autoencoder_B.pth'

loaded_model_auto_A = Autoencoder()
loaded_model_auto_B = Autoencoder()

loaded_model_auto_A.load_state_dict(torch.load(fileA))  # it takes the loaded dictionary, not the path file itself
loaded_model_auto_A.eval()
loaded_model_auto_B.load_state_dict(torch.load(fileB))  # it takes the loaded dictionary, not the path file itself
loaded_model_auto_B.eval()

# encoder of A, decoder of B

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

criterion = nn.MSELoss()

model = ConcatAutoencoder(loaded_model_auto_A.encoder, loaded_model_auto_B.decoder).to('cuda')

losses = []
sum_loss = 0
for (img, _) in tqdm(data_loader):
    img = img.to('cuda')
    img = img.reshape(-1, 28 * 28)
    recon = model(img)
    loss = criterion(recon, img)
    losses.append((loss, recon, img))
    sum_loss = sum_loss + loss.item()

print(f'average loss:{sum_loss / len(losses) :.4f}')
