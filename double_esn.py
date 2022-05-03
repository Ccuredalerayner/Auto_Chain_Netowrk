import gc

import matplotlib.pyplot as plt
import now as now
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from tqdm import tqdm
from datetime import datetime
import datasets


def double_esn_testing(feedback=False, load=None, load_b=None, train=False, gift_data=None,
                       data_size=1000, washout=[0], hidden_size=30, input_size_a=1,
                       input_size_b=2, output_size_a=2, output_size_b=3, loss_fcn=torch.nn.MSELoss(),
                       c=0.1, d=0.1, e=0.1, c_change=0.001, d_change=0.001, e_change=0.001):
    """
    :param feedback: Bool True, feedback loops on both networks. False, no feedback loops
    :param load: String the file name of the first ESN network
    :param load_b: String the file name of the second ESN network
    :param train: Bool True, the network will be trained. False, the network will not be trained
    :param gift_data: list [0] = y data, [1] = a,b data, [2] = c,d,e data
    :param data_size: int data points in dataset
    :param washout: [int] randomly removes some datapoints
    :param hidden_size: int hidden size of the network
    :param input_size_a: int input size for first network
    :param input_size_b: int input size for second network
    :param output_size_a: int output size for first network
    :param output_size_b: int output size for second network
    :param loss_fcn: pytorch loss function for calculation loss and training
    :param c: float dataset parameter
    :param d: float dataset parameter
    :param e: float dataset parameter
    :param c_change: float dataset parameter
    :param d_change: float dataset parameter
    :param e_change: float dataset parameter
    :return: loss: float total loss of the hole network
    :return: file_name_a: string file name for first ESN
    :return: file_name_b: string file name for second ESN
    """
    save_input_size_a = input_size_a
    save_input_size_b = input_size_b
    save_output_size_a = output_size_a
    save_output_size_b = output_size_b
    device = torch.device('cuda')
    dtype = torch.double
    torch.set_default_dtype(dtype)

    loss_fcn = loss_fcn

    if feedback:
        input_size_a = input_size_a + hidden_size
        input_size_b = input_size_b + hidden_size

    # load file
    if load is not None:
        file_name_a = load  # file_name = "saves/" + load
        file_name_b = load_b

        model_esn_a = ESN(input_size_a, hidden_size, output_size_a, output_steps='mean',
                          readout_training='cholesky', ).to(device)
        model_esn_a.load_state_dict(torch.load(file_name_a))  # it takes the loaded dictionary, not the path file itself
        model_esn_a.eval()

        model_esn_b = ESN(input_size_b, hidden_size, output_size_b, output_steps='mean',
                          readout_training='cholesky', ).to(device)
        model_esn_b.load_state_dict(torch.load(file_name_b))  # it takes the loaded dictionary, not the path file itself
        model_esn_b.eval()

    else:
        model_esn_a = ESN(input_size_a, hidden_size, output_size_a).to(device)
        model_esn_b = ESN(input_size_b, hidden_size, output_size_b).to(device)

    # Data
    if gift_data is not None:
        X_data, Y_data, Z_data = gift_data[0], gift_data[1], gift_data[2]
        data_size = len(X_data)
    else:
        X_data, Y_data, Z_data = datasets.sin_link(data_size, c, d, e, c_change, d_change, e_change)
    X_data = torch.tensor(X_data).type(torch.DoubleTensor).to(device)
    Y_data = torch.tensor(Y_data).type(torch.DoubleTensor).to(device)
    Z_data = torch.tensor(Z_data).type(torch.DoubleTensor).to(device)
    X_data = X_data.reshape(-1, 1, save_input_size_a)
    Y_data = Y_data.reshape(-1, 1, save_input_size_b)
    Z_data = Z_data.reshape(-1, 1, save_output_size_b)
    A_input = X_data
    B_input = Y_data

    if feedback:
        A_input = F.pad(input=X_data, pad=(0, 30, 0, 0), mode='constant', value=0)
        B_input = F.pad(input=Y_data, pad=(0, 30, 0, 0), mode='constant', value=0)

    guess_flat_a = utils.prepare_target(Y_data.clone(), [X_data.size(0)], washout)
    guess_flat_b = utils.prepare_target(Z_data.clone(), [Y_data.size(0)], washout)

    if (load is None) or train:
        # train a and b separately
        # for hidden layer pass
        hidden_layer_a = torch.zeros(1, 1, input_size_a).to(device)
        for data, answer in tqdm(zip(A_input, guess_flat_a)):
            # A
            data = data.reshape(1, 1, -1)
            # for hidden layer pass
            if feedback:
                data = torch.add(data, hidden_layer_a)
            answer = answer.reshape(1, -1)

            model_esn_a(data, washout, None, answer)  # check if None changes hidden output
            model_esn_a.fit()

            # for hidden layer pass
            if feedback:
                _, hidden_a = model_esn_a(data, washout)
                hidden_layer_a = F.pad(input=hidden_a, pad=(1, 0, 0, 0), mode='constant', value=0)

        # for hidden layer pass
        hidden_layer_b = torch.zeros(1, 1, input_size_b).to(device)
        for data, answer in tqdm(zip(B_input, guess_flat_b)):
            # B
            data = data.reshape(1, 1, -1)
            # for hidden layer pass
            if feedback:
                data = torch.add(data, hidden_layer_b)
            answer = answer.reshape(1, -1)

            model_esn_b(data, washout, None, answer)  # check if None changes hidden output
            model_esn_b.fit()

            # for hidden layer pass
            if feedback:
                _, hidden_b = model_esn_b(data, washout)
                hidden_layer_b = F.pad(input=hidden_b, pad=(save_input_size_b, 0, 0, 0), mode='constant', value=0)

    # error
    loss = 1
    if feedback:
        i = 0
        output_b_save = torch.Tensor(len(X_data), 1, save_output_size_b).to(device)
        hidden_layer_a = torch.zeros(1, 1, A_input.size(2)).to(device)
        hidden_layer_b = torch.zeros(1, 1, B_input.size(2)).to(device)
        for data, answer_a, answer_b in zip(X_data, guess_flat_a, guess_flat_b):
            data = data.reshape(1, 1, -1)
            data = torch.add(data, hidden_layer_a)

            # run datapoint on esn_a
            output_a, hidden_a = model_esn_a(data, washout)
            hidden_layer_a = F.pad(input=hidden_a, pad=(1, 0, 0, 0), mode='constant', value=0)

            output_a = F.pad(input=output_a, pad=(0, hidden_size, 0, 0), mode='constant', value=0)
            output_a_new = torch.add(output_a, hidden_layer_b)

            output_b, hidden_b = model_esn_b(output_a_new, washout)
            hidden_layer_b = F.pad(input=hidden_b, pad=(save_input_size_b, 0, 0, 0), mode='constant', value=0)

            output_b_save[i] = output_b
            i += 1

        loss = loss_fcn(output_b_save, Z_data[washout[0]:]).item()
    else:
        output_a, hidden = model_esn_a(X_data, washout)
        output_b, hidden = model_esn_b(output_a, washout)
        loss = loss_fcn(output_b, Z_data[washout[0]:]).item()

    if load is None:
        file_exe = ''
        if feedback:
            file_exe = 'F'
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        file_name_a = "saves/ESN_" + 'A_2_esn' + file_exe + '_' + date + ".pth"
        torch.save(model_esn_a.state_dict(), file_name_a)
        file_name_b = "saves/ESN_" + 'B_2_esn' + file_exe + '_' + date + ".pth"
        torch.save(model_esn_b.state_dict(), file_name_b)

    return loss, file_name_a, file_name_b

# loss , file_a, file_b = double_esn_testing(feedback=True)
# print(f'feedback is true: {loss}')
# print(f'feedback is true: {double_esn_testing(feedback=True,load=file_a,load_b=file_b)}')
