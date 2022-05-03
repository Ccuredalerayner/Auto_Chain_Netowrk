import random
import time
import datasets
import double_esn
import standard_esn
import Mega_ESN_clean
import torch as torch
import pandas as pd
import csv
import matplotlib.pyplot as plt


def test_main(number_datasets, length_dataset=100, dataset_function=datasets.multi_dataset,
              input_size=1, output_size=1, inner_size=1):
    '''
    All seven networks are trained on the same random dataset generated
    by datasets.multi_dataset. Once trained the networks are tested on
    a randomly generated test dataset using he same multi_dataset function.
    This test step is repeated number_datasets times. The outputs of
    each network are appended to a csv file defined below the function.
    :param number_datasets: int number of test sets to be generated
    :return:
    '''
    test_main_start = time.time()
    # train
    # gen data params
    # gift_data_train = datasets.multi_dataset(1000)
    gift_data_train = dataset_function(length_dataset, 1)
    loss_function = torch.nn.MSELoss()

    print('training...')
    # mega - A B A
    loss_megaABA, file_name_megaABA, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
        feedback=True, load=None, train=True, gift_data=gift_data_train, washout=[0], hidden_size=30, input_size_a=input_size,
        input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)
    row = [file_name_megaABA, '', 'mega_ABA', number_datasets, 'yes', 'train', loss_function, loss_megaABA]
    print(row)
    writer.writerow(row)

    # standard - 30 no feed
    loss, file_name_standard_30_no_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=30, input_size=input_size,
                                                                          output_size=output_size, load=None,
                                                                          loss_f=loss_function, feedback=False,
                                                                          train=True, gift_data=gift_data_train)
    row = [file_name_standard_30_no_feed, '', 'standard_30', number_datasets, 'no', 'train', loss_function, loss]
    print(row)
    writer.writerow(row)
    # standard - 30 feed
    loss, file_name_standard_30_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=30, input_size=input_size,
                                                                       output_size=output_size, load=None,
                                                                       loss_f=loss_function, feedback=True,
                                                                       train=True, gift_data=gift_data_train)

    row = [file_name_standard_30_feed, '', 'standard_30', number_datasets, 'yes', 'train', loss_function, loss]
    print(row)
    print(file_name_standard_30_feed)
    writer.writerow(row)
    # standard - 60 no feed
    loss, file_name_standard_60_no_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=60, input_size=input_size,
                                                                          output_size=output_size, load=None,
                                                                          loss_f=loss_function, feedback=False,
                                                                          train=True, gift_data=gift_data_train)

    row = [file_name_standard_60_no_feed, '', 'standard_60', number_datasets, 'no', 'train', loss_function, loss]
    print(row)
    writer.writerow(row)
    # standard - 60 feed
    loss, file_name_standard_60_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=60, input_size=input_size,
                                                                       output_size=output_size, load=None,
                                                                       loss_f=loss_function, feedback=True,
                                                                       train=True, gift_data=gift_data_train)

    row = [file_name_standard_60_feed, '', 'standard_60', number_datasets, 'yes', 'train', loss_function, loss]
    print(row)
    writer.writerow(row)
    # double - 30 no feed
    loss, file_name_double_A_30_no_feed, file_name_double_B_30_no_feed = double_esn.double_esn_testing(feedback=False,
        load=None, load_b=None, train=True, gift_data=gift_data_train, washout=[0], hidden_size=30, input_size_a=input_size,
        input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)

    row = [file_name_double_A_30_no_feed, file_name_double_B_30_no_feed, 'double_30', number_datasets, 'no', 'train',
           loss_function, loss]
    print(row)
    writer.writerow(row)
    # double - 30 feed
    loss, file_name_double_A_30_feed, file_name_double_B_30_feed = double_esn.double_esn_testing(feedback=True,
        load=None, load_b=None, train=True, gift_data=gift_data_train, washout=[0], hidden_size=30, input_size_a=input_size,
        input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)
    row = [file_name_double_A_30_feed, file_name_double_B_30_feed, 'double_30', number_datasets, 'yes', 'train', loss_function, loss]
    print(row)
    writer.writerow(row)

    #################################################################################################
    # test
    print('testing...')
    for p in range(number_datasets):
        gift_data_test = dataset_function(length_dataset, 1)
        # mega - A B A
        loss_megaABA, file_name_megaABA, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
            feedback=True, load=file_name_megaABA, train=False, gift_data=gift_data_test, washout=[0], hidden_size=30, input_size_a=input_size,
            input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)
        row = [file_name_megaABA, '', 'mega_ABA', number_datasets, 'yes', 'test', loss_function, loss_megaABA]
        writer.writerow(row)
        print(row)

        # standard - 30 no feed
        loss, file_name_standard_30_no_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=30, input_size=input_size,
                                                                              output_size=output_size, load=file_name_standard_30_no_feed,
                                                                              loss_f=loss_function, feedback=False, train=False, gift_data=gift_data_test)

        row = [file_name_standard_30_no_feed, '', 'standard_30', number_datasets, 'no', 'test', loss_function, loss]
        print(row)
        writer.writerow(row)

        # standard - 30 feed
        loss, file_name_standard_30_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=30, input_size=input_size,
                                                                           output_size=output_size, load=file_name_standard_30_feed,
                                                                           loss_f=loss_function, feedback=True, train=False, gift_data=gift_data_test)

        row = [file_name_standard_30_feed, '', 'standard_30', number_datasets, 'yes', 'test', loss_function, loss]
        print(row)
        writer.writerow(row)

        # standard - 60 no feed
        loss, file_name_standard_60_no_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=60, input_size=input_size,
                                                                              output_size=output_size, load=file_name_standard_60_no_feed,
                                                                              loss_f=loss_function, feedback=False, train=False, gift_data=gift_data_test)

        row = [file_name_standard_60_no_feed, '', 'standard_60', number_datasets, 'no', 'test', loss_function, loss]
        print(row)
        writer.writerow(row)
        # standard - 60 feed
        loss, file_name_standard_60_feed = standard_esn.single_esn_testing(washout=[0], hidden_size=60, input_size=input_size,
                                                                           output_size=output_size, load=file_name_standard_60_feed,
                                                                           loss_f=loss_function, feedback=True, train=False, gift_data=gift_data_test)

        row = [file_name_standard_60_feed, '', 'standard_60', number_datasets, 'yes', 'test', loss_function, loss]
        print(row)
        writer.writerow(row)
        # double - 30 no feed
        loss, file_name_double_A_30_no_feed, file_name_double_B_30_no_feed = double_esn.double_esn_testing(
            feedback=False, load=file_name_double_A_30_no_feed, load_b=file_name_double_B_30_no_feed,
            train=False, gift_data=gift_data_test, washout=[0], hidden_size=30, input_size_a=input_size,
            input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)
        row = [file_name_double_A_30_no_feed, file_name_double_B_30_no_feed, 'double_30', number_datasets, 'no', 'test', loss_function,
               loss]
        print(row)
        writer.writerow(row)
        # double - 30 feed
        loss, file_name_double_A_30_feed, file_name_double_B_30_feed = double_esn.double_esn_testing(
            feedback=True, load=file_name_double_A_30_feed, load_b=file_name_double_B_30_feed,
            train=False, gift_data=gift_data_test, washout=[0], hidden_size=30, input_size_a=input_size,
            input_size_b=inner_size, output_size_a=inner_size, output_size_b=output_size, loss_fcn=loss_function)
        row = [file_name_double_A_30_feed, file_name_double_B_30_feed, 'double_30', number_datasets, 'yes', 'test', loss_function,
               loss]
        print(row)
        writer.writerow(row)
        print("test_main ended in", time.time() - test_main_start, "seconds.")


# run main
# open the file in the write mode
import os

file = 'dis_saves/Final_test/test_1.csv'
os.makedirs(os.path.dirname(file), exist_ok=True)
f = open(file, 'w+')
# create the csv writer
writer = csv.writer(f)
col = ['file name A', 'file name B', 'esn type', 'number datasets', 'feedback', 'test/train', 'loss_function', 'loss']
# write a row to the csv file
writer.writerow(col)

start = time.time()

# no_datasets = 10
# length_datasets = 100
# no_data_points = 50

# testing values
no_tests_per_network = 50
length_datasets = 1000
no_networks = 50

for j in range(no_networks):
    print(j)
    print('Testing Pattern in Pattern')
    test_main(no_tests_per_network, length_datasets, dataset_function=datasets.multi_dataset, input_size=1, inner_size=2, output_size=3)
    print(j)
    print('Testing Random')
    test_main(no_tests_per_network, length_datasets, dataset_function=datasets.multi_random_dataset, input_size=1, inner_size=2, output_size=3)
    print(j)
    print('Testing Narma10 Pattern')
    test_main(no_tests_per_network, length_datasets, dataset_function=datasets.narma10_formatted, input_size=1, inner_size=1, output_size=1)
    print(j)
    print('Testing Simple Pattern')
    test_main(no_tests_per_network, length_datasets, dataset_function=datasets.multi_single_patten_dataset, input_size=1, inner_size=1, output_size=1)


print("program ended in", time.time() - start, "seconds.")

# close the file
f.close()

save = pd.read_csv(file, sep=',')
print(save)
