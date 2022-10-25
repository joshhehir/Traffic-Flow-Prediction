"""
Definition of NN model
"""
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, SimpleRNN
from keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.
    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def _get_sae2(x_train, hidden, output):
    """SAE(Auto-Encoders) TODO REPLACE THIS
    Build SAE Model.
    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    # batch_size = 64
    input_dim = x_train[0].shape[0]
    learning_rate = 1e-5

    model = Sequential()

    # first layer
    model.add(Dense(hidden, input_dim=input_dim, name='input', activation="relu",
                    activity_regularizer=regularizers.l1(learning_rate)))

    # second layer
    model.add(Dense(hidden / 2, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Third layer
    model.add(Dense(hidden / 4, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Fourth/Bottleneck layer
    model.add(Dense(hidden / 8, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Fifth layer
    model.add(Dense(hidden / 4, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # add dropout
    model.add(Dropout(0.2))

    # Sixth layer
    model.add(Dense(hidden / 2, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Output layer
    model.add(Dense(output, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate)))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.
    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models


def get_saes2(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.
    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[2], layers[0])
    sae2 = _get_sae(layers[0] * 2, layers[2], layers[0] * 2)
    sae3 = _get_sae(layers[0] * 4, layers[2], layers[-1])

    """saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))
    """
    models = [sae1, sae2, sae3]

    return models


def get_srnn(units):
    """SRNN(Simple recurrent neural network)
    Build SRNN Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(SimpleRNN(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model
