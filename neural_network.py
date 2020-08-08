"""
 *
 * project No. 4 Artificial Intelligence
 * description: implementation and evaluation of neutral network by back-propagation algorithm
 * The effect of depth and width of hidden layers on prediction, have been investigated and the
 * results are plotted
 * @author  Ramtin Hosseni
 * @submitted  05.02.2018
 *
"""

from math import exp
from random import seed
import random
import numpy as np
import matplotlib.pyplot as plt
import os

seed(1)

LEARNING_RATE = 0.1
ITERATION = 100


# setting the env
def set_environment():
    if not os.path.exists("./output/"):
        os.makedirs("./output/")


# plotting
def plotting_curves(x_label, y_label, title, x_values, y_values):
    plt.plot(x_values, y_values, 'bo', label='error_rate')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('./output/' + title + '.png')
    plt.close()


# function that we later need to compare by using one-hot encoding
def set_highest_bit_in_prediction_to_one(predicted_output):
    prediction_with_highest_bit = [0 for i in range(len(predicted_output))]
    max = 0
    for idx, value in enumerate(predicted_output):
        if value > max:
            max = value
            index = idx
    prediction_with_highest_bit[index] = 1
    return prediction_with_highest_bit


# one-hot conversion
def convert_to_one_hot(number, length_one_hot):
    one_hot_num = [0 for i in range(0, length_one_hot)]
    one_hot_num[number] = 1
    return one_hot_num


# determine list of labels in a given file in the file
def determine_output_nodes(line):
    list_output_nodes = []
    for each in line.split(','):
        list_output_nodes.append(each)
    return list_output_nodes


# it is the parser of the file
def return_train_test_data_with_labels(file_name):
    with open(file_name, 'r') as input_file:
        list_train_test_data_with_labels = []
        for line in input_file:
            if not (line.startswith('\n')):
                list_train_test_data_with_labels.append(line.strip('\n').split(','))
    for each in list_train_test_data_with_labels:
        if each[-1] == 'Iris-setosa':
            each[-1] = 0
        elif each[-1] == 'Iris-versicolor':
            each[-1] = 1
        else:
            each[-1] = 2

    for i in range(len(list_train_test_data_with_labels)):
        for j in range(len(list_train_test_data_with_labels[i])):
            list_train_test_data_with_labels[i][j] = float(list_train_test_data_with_labels[i][j])
    return list_train_test_data_with_labels


# set up the network
def initialize_network(n_inputs, n_hidden, n_outputs, depth=None):
    network = list()
    if depth == 0:
        output_layer = [{'weights': [random.uniform(-0.1, 0.1) for _ in range(n_inputs + 1)]} for _ in range(n_outputs)]
        network.append(output_layer)
    else:
        for i in range(0, depth):
            if i == 0:
                hidden_layer = [{'weights': [random.uniform(-0.1, 0.1) for _ in range(n_inputs + 1)]}
                                for _ in range(n_hidden)]
                network.append(hidden_layer)
            else:
                hidden_layer = [{'weights': [random.uniform(-0.1, 0.1) for _ in range(n_hidden + 1)]}
                                for _ in range(n_hidden)]
                network.append(hidden_layer)

        output_layer = [{'weights': [random.uniform(-0.1, 0.1) for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
        network.append(output_layer)
    return network


# activate function for determining the value of each node
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# sigmoid function
def sigmoid(activation):
    if activation < -50.0:
        activation = -50.0
    return 1.0 / (1.0 + exp(-activation))


# forward propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# sigmoid prime function
def transfer_derivative(output):
    return output * (1.0 - output)


# backward algorithm
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# update the weight of each nodes when running back propagation algorithm by using delta and weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# training network by using forward/ backward algorithm
def learn_network(network, train, l_rate, n_epoch, n_outputs, test_set=None):
    error_rate_train = []
    error_rate_test = []
    for epoch in range(n_epoch):
        count_num_error_train = 0
        count_num_error_test = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = convert_to_one_hot(int(row[-1]), n_outputs)  # one_hot encoding
            predicted_with_highest_bit_set_to_one = set_highest_bit_in_prediction_to_one(outputs)
            if predicted_with_highest_bit_set_to_one != expected:
                count_num_error_train += 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        error_rate_train.append(count_num_error_train / len(train))
        # print('>epoch=%d, count_num_error_train=%.d' % (epoch, count_num_error_train))
        if test_set is not None:
            for row_test in test_set:
                output_test = forward_propagate(network, row_test)
                expected_test = convert_to_one_hot(int(row_test[-1]), n_outputs)  # one_hot encoding
                predicted_with_highest_bit_set_to_one_for_test = set_highest_bit_in_prediction_to_one(output_test)
                if predicted_with_highest_bit_set_to_one_for_test != expected_test:
                    count_num_error_test += 1
            error_rate_test.append(count_num_error_test / len(test_set))
            print('>error_rate_test=%f' % (count_num_error_test / len(test_set)))
            print('>epoch=%d, count_num_error_test=%.d' % (epoch, count_num_error_test))
    return error_rate_test[-1]


# partition data into train, test
def partition_data_into_train_test(data_list):
    list_labels = np.unique([each[-1] for each in data_list])
    list_data_in_each_class = [[] for _ in range(len(list_labels))]
    list_data_in_each_class[0] = np.random.permutation(data_list[0:50]).tolist()
    list_data_in_each_class[1] = np.random.permutation(data_list[50:100]).tolist()
    list_data_in_each_class[2] = np.random.permutation(data_list[100:150]).tolist()

    train_list = list_data_in_each_class[0][0:40] + list_data_in_each_class[1][0:40] + list_data_in_each_class[2][0:40]
    test_list = list_data_in_each_class[0][40:50] + list_data_in_each_class[1][40:50] + list_data_in_each_class[2][
                                                                                        40:50]
    return train_list, test_list


def main():
    set_environment()
    file_names = ['data_set/PS4 - Iris data.txt']
    list_train_test_data_with_labels = return_train_test_data_with_labels(file_names[0])
    train_list, test_list = partition_data_into_train_test(list_train_test_data_with_labels)

    n_inputs = len(train_list[0]) - 1  # number of features
    n_outputs = len(set([row[-1] for row in train_list]))  # number of labels

    depth = 1
    width_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_err_as_function_w = []
    for width in width_list:
        network = initialize_network(n_inputs, width, n_outputs, depth)
        test_err_as_function_w.append(learn_network(network, train_list, LEARNING_RATE, ITERATION, n_outputs, \
                                                    test_set=test_list))
        print("test_err_as_function_w", test_err_as_function_w)
    plotting_curves("width", "error rate",
                    "error rate after {} iteration- different width, depth = 1".format(ITERATION), \
                    width_list, test_err_as_function_w)


    width = 9
    depth_list = [0, 1, 2, 3, 4, 5]
    test_err_as_function_d = []
    for depth in depth_list:
        network = initialize_network(n_inputs, width, n_outputs, depth)
        test_err_as_function_d.append(learn_network(network, train_list, LEARNING_RATE, ITERATION, n_outputs,
                                                    test_set=test_list))
        print("test_err_as_function_d", test_err_as_function_d)
    plotting_curves("depth", "error rate",
                    "error rate after {} iteration- different depth, width = 9".format(ITERATION), \
                    depth_list, test_err_as_function_d)


if __name__ == "__main__":
    main()
