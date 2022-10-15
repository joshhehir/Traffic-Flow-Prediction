"""
Processing the data
"""
from time import gmtime, strftime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data.scats import ScatsData

SCATS_DATA = ScatsData()


def format_time_to_index(time):
    time_places = time.split(":")
    hours = int(time_places[0])
    minutes = int(time_places[1])
    total_minutes = hours * 60 + minutes

    return (0, total_minutes / 15)[total_minutes > 0]


def format_time(index):
    return strftime("%H:%M", gmtime(index * 15 * 60))


def format_date(date):
    return pd.datetime.strftime(date, "%d/%m/%Y")


def process_data(scats_number, junction, lags):
    volume_data = SCATS_DATA.get_scats_volume(scats_number, junction)
    volume_training = volume_data[:2016]
    volume_testing = volume_data[2016:]

    # scaler = StandardScaler().fit(volume.values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(volume_training.reshape(-1, 1))
    flow1 = scaler.transform(volume_training.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(volume_testing.reshape(-1, 1)).reshape(1, -1)[0]

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

    return x_train, y_train, x_test, y_test, scaler