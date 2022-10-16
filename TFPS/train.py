"""
Train the NN model.
"""
import sys
import warnings
import argparse
import os
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from data.scats import ScatsData
from settings import get_setting

warnings.filterwarnings("ignore")
SCATS_DATA = ScatsData()


def train_model(model, x_train, y_train, name, scats, junction, config):
    """Trains a single model
    Parameters needed
        - model: the model to be trained
        - x_train: the input data for training
        - y_train: the output or result from training
        - name: the name of the model
        - scats: the scats site number
        - junction: the number that corresponds to the scat site based on the vic roads data
        - config: values for training found in config.json """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        x_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    folder = "model/{0}/{1}".format(name, scats)
    file = "{0}/{1}".format(folder, junction)

    if not os.path.exists(folder):
        os.makedirs(folder)

    model.save("{0}.h5".format(file))

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("{0} loss.csv".format(file), encoding='utf-8', index=False)
    print("Training complete!")


def train_seas(models, x_train, y_train, name, scats, junction, config):
    """Trains the SAEs model
    Parameters needed:
        - model: list type of SAE model to be trained
        - x_train: the input data for training
        - y_train: the output or result from training
        - name: the name of the model
        - scats: the scats site number
        - junction: the number that corresponds to the scat site based on the vic roads data
        - config: values for training found in config.json """

    temp = x_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)
        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)
        models[i] = m
    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)
    train_model(saes, x_train, y_train, name, scats, junction, config)


def train_with_args(scats, junction, model_to_train):
    """ Begin training a model with arguments

    Parameters needed:
        - scats: the scats number
        - junction: the number that corresponds to the scat site based on the vic roads data
        - model_to_train: the NN model
        """

    scats_numbers = SCATS_DATA.get_all_scats_numbers()

    if scats != "all":
        scats_numbers = [scats]

    for scats_site in scats_numbers:
        junctions = SCATS_DATA.get_scats_approaches(scats_site)

        if junction != "all":
            junctions = [junction]

        config = get_setting("train_config")
        for junction in junctions:
            x_train, y_train, _, _, _ = process_data(scats_site, junction, config["lag"])

            if model_to_train == 'lstm':
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                m = model.get_lstm([12, 64, 64, 1])
                train_model(m, x_train, y_train, model_to_train, scats_site, junction, config)
            if model_to_train == 'gru':
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                m = model.get_gru([12, 64, 64, 1])
                train_model(m, x_train, y_train, model_to_train, scats_site, junction, config)
            if model_to_train == 'saes':
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
                m = model.get_saes([12, 400, 400, 400, 1])
                train_seas(m, x_train, y_train, model_to_train, scats_site, junction, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats",
        default="970",
        help="SCATS site number.")
    parser.add_argument(
        "--junction",
        default="1",
        help="The approach to the site.")
    parser.add_argument(
        "--model",
        default="saes",
        help="Model to train.")
    args = parser.parse_args()

    train_with_args(int(args.scats), int(args.junction), args.model)


if __name__ == '__main__':
    main(sys.argv)
