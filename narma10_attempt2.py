import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time

import NARMA10_again

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

data_size = 100

raw_data = NARMA10_again.generate_NARMA10(data_size)

X_data, Y_data = raw_data

X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)
X_data = X_data.reshape(-1, 1, 1)
Y_data = Y_data.reshape(-1, 1, 1)

trX = X_data[:data_size // 2]
trY = Y_data[:data_size // 2]
tsX = X_data[data_size // 2:]
tsY = Y_data[data_size // 2:]

washout = [30]
hidden_size = 30
input_size = 1
output_size = 1
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size)
    model.to(device)

    _, hidden = model(trX, washout, None, trY_flat)
    model.fit()

    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    output, hidden = model(tsX, [0], hidden)
    print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")
