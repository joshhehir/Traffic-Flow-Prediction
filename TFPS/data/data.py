"""
Processing the data
"""
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Node:
    def __init__(self, name, x_train, y_train, x_test, y_test, scaler):
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler


def process_data(file, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns 
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    df = pd.read_csv(file, encoding='utf-8').fillna(0)
    nodes = []
    for x in range(int(len(df) / 31)):
        # scaler = StandardScaler().fit(df1[attr].values)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(get_month_array(df, x * 31, 31).reshape(-1, 1))
        flow1 = scaler.transform(get_month_array(df, x * 31, 15).reshape(-1, 1)).reshape(1, -1)[0]
        flow2 = scaler.transform(get_month_array(df, x * 31 + 15, 16).reshape(-1, 1)).reshape(1, -1)[0]

        train, test = [], []
        for i in range(lags, len(flow1)):
            train.append(flow1[i - lags: i + 1])
        for i in range(lags, len(flow2)):
            test.append(flow2[i - lags: i + 1])

        train = np.array(train)
        test = np.array(test)
        np.random.shuffle(train)

        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]

        node = Node(df["Location"].values[x * 31], x_train, y_train, x_test, y_test, scaler)
        nodes.append(node)

        print(node.name)

    return nodes


def get_month_array(data, start_position, month_length):
    array = []
    for x in range(96):
        column = data["V{:02d}".format(x)].values
        temp = []
        for y in range(len(column)):
            temp.append(column[y])
        array.append(temp)
    reordered_array = list(zip(*array[::1]))
    output = []
    for x in range(len(reordered_array)):
        for y in range(len(reordered_array[x])):
            if x >= start_position and x < start_position + month_length:
                output.append(reordered_array[x][y])
    return np.array(output)


def reorder_array(array, spacing, length):
    output = np.array(array)
    print(len(array))
    idx = []

    for x in range(spacing):
        for y in range(length):
            idx.append((y * spacing) + x)
    return output[idx]


if __name__ == '__main__':
    process_data("2006.csv", 12)
