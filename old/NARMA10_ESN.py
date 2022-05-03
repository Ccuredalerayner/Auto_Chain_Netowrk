import torch.nn
from torchesn.nn import ESN
import time

import NARMA10

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

raw_data = NARMA10.getData(10)
inter_data = torch.tensor(raw_data[0]).reshape(-1,1)
last_data = inter_data[-1:].to(device)
x_data = inter_data[:-2].to(device)
x_data = x_data.resize(8,1,1)
y_data = inter_data[1:-1].to(device)

'''all_data = torch.tensor(NARMA10.getData(10)[0])
x_data = torch.utils.data.DataLoader(all_data[:-1])
y_data = torch.utils.data.DataLoader(all_data[1:])'''

input_size = 1
output_size = 1
hidden_size = 10
washout = [hidden_size]
loss_fcn = torch.nn.MSELoss()

start = time.time()

model = ESN(input_size, hidden_size, output_size,
            output_steps='mean', readout_training='cholesky').to(device)

# Training
output, hidden = model(x_data, washout, None, y_data)
model.fit()

# Guess
output, hidden = model(x_data)

# Test
y_test_predicted = model(y_data[-1:])
#print(loss_fcn(y_test_predicted, last_data))
