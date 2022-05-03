import gc
import now as now
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from tqdm import tqdm
from datetime import datetime

import NARMA10_again

# date and time
import datasets

date_time = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

# misc
device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

# hyperparameters
data_size = 500
washout = [0]
hidden_size = 30
input_size = 1 + hidden_size
output_size = 2
loss_fcn = torch.nn.MSELoss()

# datasets
data_type = 1

if data_type == 0:
    # DATA FORMATTING
    # narma
    X_data, Y_data = NARMA10_again.generate_NARMA10(data_size)

    X_data = torch.from_numpy(X_data).to(device)
    Y_data = torch.from_numpy(Y_data).to(device)
    X_data = X_data.reshape(-1, 1, 1)
    Y_data = Y_data.reshape(-1, 1, 1)

    # X_data = F.pad(input=X_data, pad=(0, 30, 0, 0), mode='constant', value=0)

    trX = X_data[:data_size // 2]
    trY = Y_data[:data_size // 2]
    tsX = X_data[data_size // 2:]
    tsY = Y_data[data_size // 2:]
elif data_type == 1:
    # new data
    X_data, Y_data = datasets.sin_link(data_size)
    X_data = torch.tensor(X_data).to(device)
    Y_data = torch.tensor(Y_data).to(device)
    X_data = X_data.reshape(-1, 1, 1)
    Y_data = Y_data.reshape(-1, 1, 2)

    X_data = F.pad(input=X_data, pad=(0, 30, 0, 0), mode='constant', value=0)

    trX = X_data
    trY = Y_data
    # tsX = X_data_train
    # tsY = Y_data_train


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


if __name__ == "__main__":
    start = time.time()

    # Autoencoder setup
    model_auto = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_auto.parameters(), lr=0.001)
    num_epochs = 200  # 200 gives a near always 0 loss

    # ESN setup
    model_esn = ESN(input_size, hidden_size, output_size).to(device)

    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    hidden_layer = torch.zeros(1, 1, 31).to(device)

    test = 0
    save = False

    if test == 0:  # looped: feedback loop
        for data, answer in tqdm(zip(trX, trY_flat)):
            data = data.reshape(1, 1, -1)
            data = torch.add(data, hidden_layer)
            answer = answer.reshape(1, -1)

            model_esn(data, washout, None, answer)  # check if None changes hidden output
            model_esn.fit()

            _, hidden = model_esn(data, washout)
            hidden_layer = F.pad(input=hidden, pad=(1, 0, 0, 0), mode='constant', value=0)

            # train autoencoder
            for epoch in range(num_epochs):
                recon = model_auto(hidden)
                loss = criterion(recon, hidden)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Testing loss of autoencoder
        print(f'Epoch:{epoch + 1}, loss:{loss.item():.4f}')

        # Training Test
        # ESN
        output, hidden = model_esn(trX, washout)
        print("Training error before auto:", loss_fcn(output, trY[washout[0]:]).item())

        for data, answer in zip(trX, trY_flat):
            data = data.reshape(1, 1, -1)
            data = torch.add(data, hidden_layer)
            answer = answer.reshape(1, -1)

            model_esn(data, washout, None, answer)  # check if None changes hidden output
            model_esn.fit()

            _, hidden = model_esn(data, washout)
            hidden_layer = F.pad(input=hidden, pad=(1, 0, 0, 0), mode='constant', value=0)

        # Training Test
        # ESN
        output, hidden = model_esn(trX, washout)
        print("Training error after auto:", loss_fcn(output, trY[washout[0]:]).item())

    elif test == 1:  # looped: no feedback loop
        for data, answer in zip(trX, trY_flat):
            data = data.reshape(-1, 1, 31)
            answer = answer.reshape(-1, 1)

            model_esn(data, washout, None, answer)
            model_esn.fit()

        # Training Test
        # ESN
        output, hidden = model_esn(trX, washout)
        print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    elif test == 2:  # not looped: no feedback
        model_esn(trX, washout, None, trY_flat)
        model_esn.fit()

    elif test == 3:  # loading...
        date_time_load = '_23-08-2021_17-19-21'
        file_auto_load = "saves/Autoencoder" + date_time_load + ".pth"
        file_esn_load = "saves/ESN" + date_time_load + ".pth"

        model_auto_loaded = Autoencoder().to(device)
        model_auto_loaded.load_state_dict(
            torch.load(file_auto_load))  # it takes the loaded dictionary, not the path file itself
        model_auto_loaded.eval()

        model_esn_loaded = ESN(input_size, hidden_size, output_size, output_steps='mean',
                               readout_training='cholesky').to(device)
        model_esn_loaded.load_state_dict(
            torch.load(file_esn_load))  # it takes the loaded dictionary, not the path file itself
        model_esn_loaded.eval()
        output, hidden = model_esn(tsX, washout)

    # Test Test
    # ESN
    if data_type == 0:
        output, hidden = model_esn(tsX, [0], hidden)
        print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")

    # Saving
    if save:
        file_auto = "saves/Autoencoder" + date_time + ".pth"
        file_esn = "saves/ESN" + date_time + ".pth"
        torch.save(model_auto.state_dict(), file_auto)
        torch.save(model_esn.state_dict(), file_esn)
