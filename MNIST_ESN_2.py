import torch.nn
from torchvision import datasets, transforms
from torchesn.nn import ESN

from tqdm import tqdm
import time

device = torch.device('cuda')


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
batch_size = 1
hidden_size = 30
input_size = 1 #+ hidden_size
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
model = ESN(input_size, hidden_size, output_size,
            output_steps='mean', readout_training='cholesky')
model.to(device)

# Fit the model
fit_count = 0
for batch in tqdm(train_iter):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    x = reshape_batch(x)
    target = one_hot(y, output_size)
    washout_list = [int(washout_rate * x.size(0))] * x.size(1)

    model(x, washout_list, None, target)
    model.fit()
    fit_count += 1

print(f'done fit')

# Evaluate on training set
tot_correct = 0
tot_obs = 0

eval_count = 0

for batch in tqdm(train_iter):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    x = reshape_batch(x)
    washout_list = [int(washout_rate * x.size(0))] * x.size(1)

    output, hidden = model(x, washout_list)
    tot_obs += x.size(1)
    tot_correct += loss_fcn(output[-1], y.type(torch.get_default_dtype()))
    eval_count += 1

print(f'done evaluation')
print(f'Training accuracy:{tot_correct / tot_obs}')

# Test
for batch in tqdm(test_iter):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    x = reshape_batch(x)
    washout_list = [int(washout_rate * x.size(0))] * x.size(1)

    output, hidden = model(x, washout_list)
    tot_obs += x.size(1)
    tot_correct += loss_fcn(output[-1], y.type(torch.get_default_dtype()))

print(f'Test accuracy:{tot_correct / tot_obs}')

print(f'Ended in {time.time() - start} seconds.')

# saving
file = "model.pth"
torch.save(model.state_dict(), file)

# loading
# loaded_model = ESN(input_size, hidden_size, output_size,output_steps='mean', readout_training='cholesky')
# loaded_model.load_state_dict(torch.load(file))  # it takes the loaded dictionary, not the path file itself
# loaded_model.eval()

