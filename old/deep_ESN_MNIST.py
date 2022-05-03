import torch.nn
from torchvision import datasets, transforms
from torchesn.nn import ESN

import echotorch.utils
import echotorch as echo
from echotorch.nn.reservoir import DeepESN

from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def one_hot(y, output_dim):
    onehot = torch.zeros(y.size(0), output_dim, device=y.device)

    for i in range(output_dim):
        onehot[y == i, i] = 1

    return onehot


def Accuracy_Correct(y_pred, y_true):
    labels = torch.argmax(y_pred, 1).type(y_pred.type())
    correct = len((labels == y_true).nonzero())
    return correct


def reshape_batch(batch):
    batch = batch.view(batch.size(0), batch.size(1), -1)
    return batch.transpose(0, 1).transpose(0, 2)


# hyper parameters
batch_size = 256
input_size = 1
hidden_size = 30
output_size = 10
washout_rate = 0.2

loss_fcn = Accuracy_Correct

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_iter = torch.utils.data.DataLoader(datasets.MNIST('./datasets', train=True, download=True, transform=transform),
                                         batch_size=batch_size, shuffle=True)

test_iter = torch.utils.data.DataLoader(datasets.MNIST('./datasets', train=False, transform=transform),
                                        batch_size=batch_size, shuffle=False)

start = time.time()

# Training
esn = echo.nn.reservoir.DeepESN(n_layers=2, input_dim=input_size, output_dim=output_size, hidden_dim=hidden_size)
esn.to(device)

# Fit the model
fit_count = 0
for batch in tqdm(train_iter):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    x = reshape_batch(x)
    target = one_hot(y, output_size)
    washout_list = [int(washout_rate * x.size(0))] * x.size(1)

    esn(x, washout_list, None, target)
    esn.fit()
    fit_count += 1

print(f'done fit')