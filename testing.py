import math
import random
from time import sleep
from tqdm import tqdm

import torch.nn as nn
import torch as torch
from torchvision import datasets, transforms
from torchesn.nn import ESN

'''loop = 100

for i in tqdm(range(loop)):
    sleep(1)'''
'''
batch_size = 256
input_size = 1
hidden_size = 30
output_size = 10
washout_rate = 0.2

FILE = "model.pth"
model = ESN(input_size, hidden_size, output_size,
            output_steps='mean', readout_training='cholesky')

print(model.state_dict())
loaded_model = ESN(input_size, hidden_size, output_size,output_steps='mean', readout_training='cholesky')
loaded_model.load_state_dict(torch.load(FILE))  # it takes the loaded dictionary, not the path file itself
loaded_model.eval()

print(loaded_model.state_dict())'''

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
            nn.Linear(8, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 30))

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder

#FILE = "saves/Autoencoder_A_31-08-2021_16-54-48.pth"

#loaded_model = Autoencoder()
#loaded_model.load_state_dict(torch.load(FILE))  # it takes the loaded dictionary, not the path file itself
#loaded_model.eval()

#print(loaded_model.state_dict())

for x in range(20):
    print((0.1 * random.randrange(1,10,1)))