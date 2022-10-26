"""
Train the NN model.
"""
import sys
import warnings
import argparse
import os
import json
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from data.scats import ScatsData
from model.model import get_sae
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


def train_saes(x_train, name, scats, junction, config):
    """Trains the SAEs model TODO FIX THIS
    Parameters needed:
        - model: list type of SAE model to be trained
        - x_train: the input data for training
        - y_train: the output or result from training
        - name: the name of the model
        - scats: the scats site number
        - junction: the number that corresponds to the scat site based on the vic roads data
        - config: values for training found in config.json """

    temp = x_train

    autoencoder_1 = get_sae(temp, 400, 96)
    autoencoder_1.compile(loss="mse", optimizer="adam", metrics=['mape'])
    stack_1 = autoencoder_1.fit(x_train, x_train, batch_size=config["batch"], epochs=config["epochs"],
                                validation_split=0.33)

    autoencoder_2_input = autoencoder_1.predict(temp)
    # autoencoder_2_input = np.concatenate(autoencoder_2_input, x_train, axis=1)
    # autoencoder_2_input = np.append(autoencoder_2_input, x_train, axis=1)

    autoencoder_2 = get_sae(autoencoder_2_input, 400, 96)
    autoencoder_2.compile(loss="mse", optimizer="adam", metrics=['mape'])
    stack_2 = autoencoder_2.fit(autoencoder_2_input, autoencoder_2_input, batch_size=config["batch"],
                                epochs=config["epochs"], validation_split=0.33)

    autoencoder_3_input = autoencoder_2.predict(autoencoder_2_input)
    # autoencoder_3_input = np.append(autoencoder_3_input, autoencoder_2_input, axis=1)

    autoencoder_3 = get_sae(autoencoder_3_input, 400, 96)
    autoencoder_3.compile(loss="mse", optimizer="adam", metrics=['mape'])
    stack_3 = autoencoder_3.fit(autoencoder_3_input, autoencoder_3_input, batch_size=config["batch"],
                                epochs=config["epochs"], validation_split=0.33)

    folder = "model/saes/{1}".format(name, scats)
    file = "{0}/{1}".format(folder, junction)

    temp1 = autoencoder_3.predict(x_train)
    if not os.path.exists(folder):
        os.makedirs(folder)

    autoencoder_3.save("{0}.h5".format(file))

    df = pd.DataFrame.from_dict(stack_3.history)
    df.to_csv("saes loss.csv".format(file), encoding='utf-8', index=False)
    print("Training complete!")

    return autoencoder_3


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
            print("training {0} : {1} - {2}".format(model_to_train, scats_site, junction))
            try:
                x_train, y_train, x_test, y_test, scaler = process_data(scats_site, junction, config["lag"])

                if model_to_train == 'lstm':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = model.get_lstm([96, 64, 64, 1])
                    train_model(m, x_train, y_train, model_to_train, scats_site, junction, config)
                if model_to_train == 'gru':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = model.get_gru([96, 64, 64, 1])
                    train_model(m, x_train, y_train, model_to_train, scats_site, junction, config)
                if model_to_train == 'saes':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = train_saes(x_train, model_to_train, scats_site, junction, config)
                    # def train_seas2(x_train, name, scats, junction, config):
                if model_to_train == 'srnn':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = model.get_srnn([96, 64, 64, 1])
                    train_model(m, x_train, y_train, model_to_train, scats_site, junction, config)

                m.summary()
                predicted = m.predict(x_test)
                predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                with open('predictedvalues.json', 'r') as openfile:

                    # Reading from json file
                    json_object = json.load(openfile)
                scats_site = str(scats_site)
                junction = str(junction)
                try:
                    print(json_object[model_to_train])
                except:
                    json_object[model_to_train] = {}
                try:
                    print(json_object[model_to_train][scats_site])
                except:
                    json_object[model_to_train][scats_site] = {}
                json_object[model_to_train][scats_site][junction] = predicted.tolist()
                json_object = json.dumps(json_object, indent=4)
                with open("predictedvalues.json", "w") as outfile:
                    outfile.write(json_object)
            except:
                print("Could not create model for {0} : {1} - {2}".format(model_to_train, scats_site, junction))


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
