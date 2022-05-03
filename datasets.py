import random
import matplotlib.pyplot as plt
import math

import NARMA10_again


def random_data(n, start=0.0, stop=1.0):
    """
    :param n: int Length of dataset
    :param start: float Small boundary for random number
    :param stop: float Large boundary for random number
    :return: data_y: [float] y values for dataset
    :return: data_ab: [[float]] a, b values for dataset
    :return: data_cde: [[float]] c, d, e values for dataset
    """
    data_y = [float(round(random.uniform(start, stop), 5)) for x in range(n)]
    data_ab = [[float(round(random.uniform(start, stop), 5)), float(round(random.uniform(start, stop), 5))] for x in range(n)]
    data_cde = [[float(round(random.uniform(start, stop), 5)),
                 float(round(random.uniform(start, stop), 5)), float(round(random.uniform(start, stop), 5))] for x in range(n)]
    # data_ab = [round(random.uniform(start, stop), 5) for x in range(n)]
    # data_cde = [round(random.uniform(start, stop), 5) for x in range(n)]

    return data_y, data_ab, data_cde


def single_patten_dataset(n, repete=10):
    """
    :param n: int Length of dataset
    :param repete: int Length of repeated loop
    :return: data_y: [float] y values for dataset
    :return: data_cde: [[float]] c, d, e values for dataset
    """
    data_cde = []

    sequence = [float(random.randint(0, 100)) for x in range(repete)]

    data_y = [x for x in range(n)]
    for y in range(n):
        try:
            value = y % len(sequence)
        except ZeroDivisionError:
            value = 0
        data_cde.append(sequence[value])

    return data_y, data_y, data_cde


def sin_link(n, c=0.1, d=0.1, e=0.1, c_change=0.001, d_change=0.001, e_change=0.001):
    """
    :param n: int Number of generated dataset values
    :param c: float initial starting value
    :param d: float initial starting value
    :param e: float initial starting value
    :param c_change: float time step change for c param
    :param d_change: float time step change for c param
    :param e_change: float time step change for c param
    :return: data_y: [float] y values for dataset
    :return: data_ab: [[float]] a, b values for dataset
    :return: data_cde: [[float]] c, d, e values for dataset
    """
    data_y = []
    data_ab = []
    data_cde = []
    for i in range(n):
        theta = i * 0.01

        c = c + c_change
        d = d + d_change
        e = e + e_change

        a_new = c ** 2 + d
        b_new = e ** 2 + d

        y = a_new * math.sin(theta * b_new)

        data_y.append(y)
        data_ab.append([a_new, b_new])
        data_cde.append([c, d, e])

    return data_y, data_ab, data_cde


def multi_dataset(n, number_of_sets=1):
    '''
    :param n: int Number of generated dataset values
    :param number_of_sets: int number of datasets in output
    :return: y_list: [float] y values for datasets
    :return: ab_list: [[float]] a, b values for datasets
    :return: cde_list: [[float]] c, d, e values for datasets
    '''
    y_list = []
    ab_list = []
    cde_list = []
    for i in range(number_of_sets):
        c = 0.1 * random.randrange(1, 9, 1)
        d = 0.1 * random.randrange(1, 9, 1)
        e = 0.1 * random.randrange(1, 9, 1)
        c_change = 0.001  # * random.randrange(1, 9, 1)
        d_change = 0.001  # * random.randrange(1, 9, 1)
        e_change = 0.001  # * random.randrange(1, 9, 1)
        y_new, ab_new, cde_new = sin_link(n, c, d, e, c_change, d_change, e_change)
        y_list += y_new
        ab_list += ab_new
        cde_list += cde_new
    return y_list, ab_list, cde_list

def multi_random_dataset(n, number_of_sets=1):
    """
    :param n: int Number of generated dataset values
    :param number_of_sets: int number of datasets in output
    :return: y_list: [float] y values for datasets
    :return: ab_list: [[float]] a, b values for datasets
    :return: cde_list: [[float]] c, d, e values for datasets
    """
    y_list = []
    ab_list = []
    cde_list = []
    for x in range(number_of_sets):
        y_new, ab_new, cde_new = random_data(n, start=0.0, stop=5.0)
        y_list.append(y_new)
        ab_list.append(ab_new)
        cde_list.append(cde_new)
    return y_list, ab_list, cde_list


def multi_single_patten_dataset(n, number_of_sets=1):
    """
    :param n: int Number of generated dataset values
    :param number_of_sets: int number of datasets in output
    :return: y_list: [float] y values for datasets
    :return: ab_list: [[float]] a, b values for datasets
    :return: cde_list: [[float]] c, d, e values for datasets
    """
    y_list = []
    ab_list = []
    cde_list = []
    for x in range(number_of_sets):
        y_new, ab_new, cde_new = single_patten_dataset(n, repete=10)
        y_list.append(y_new)
        ab_list.append(ab_new)
        cde_list.append(cde_new)
    return y_list, ab_list, cde_list


def narma10_formatted(n, number_of_sets=1):
    y_list = []
    ab_list = []
    cde_list = []
    for x in range(number_of_sets):
        inputs, outputs = NARMA10_again.generate_NARMA10(n)
        y_list.append(inputs)
        ab_list.append(inputs)
        cde_list.append(outputs)
    return y_list, ab_list, cde_list