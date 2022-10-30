"""
Definition of NN model
"""
import keras.layers
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    # Create a Sequential model
    model = Sequential()

    # Add the first LSTM Layer with unit count based on unit's input param
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))

    # Add the second LSTM layer with unit count based on unit's hidden param
    model.add(LSTM(units[2]))

    # Add a Dropout layer of 0.2
    model.add(Dropout(0.2))

    # Add the output layer with unit's output param
    model.add(Dense(units[3], activation='sigmoid'))

    # Return the model from function scope
    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    # Create a Sequential model
    model = Sequential()

    # Add the first GRU Layer with unit count based on unit's input param
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))

    # Add the second GRU Layer with unit count based on unit's input param
    model.add(GRU(units[2]))

    # Add a Dropout layer of 0.2
    model.add(Dropout(0.2))

    # Add the output layer with unit's output param
    model.add(Dense(units[3], activation='sigmoid'))

    # Return the model from function scope
    return model


def get_sae(x_train, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model, uses LeakyReLU as the activation function.
    # Arguments
        x_train: Array, the input data for training
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    # Get the input dimensions
    input_dim = x_train[0].shape[0]

    # Create a Sequential model
    model = Sequential()

    # First dense layer
    model.add(Dense(hidden, input_dim=input_dim, name='input', activation=keras.layers.LeakyReLU(alpha=0.1)))

    # Second dense layer
    model.add(Dense(hidden / 2, activation=keras.layers.LeakyReLU(alpha=0.1)))

    # Third dense layer
    model.add(Dense(hidden / 4, activation=keras.layers.LeakyReLU(alpha=0.1)))

    # Fourth dense layer
    model.add(Dense(hidden / 8, activation=keras.layers.LeakyReLU(alpha=0.1)))

    # Add a Dropout layer of 0.2
    model.add(Dropout(0.2))

    # Fifth dense layer
    model.add(Dense(hidden / 4, activation="sigmoid"))

    # Sixth dense layer
    model.add(Dense(hidden / 2, activation="sigmoid"))

    # Output layer
    model.add(Dense(output, activation="sigmoid"))

    # Return the model from function scope
    return model


def get_srnn(units):
    """SRNN(Simple recurrent neural network)
    Build SRNN Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    # Create a Sequential model
    model = Sequential()

    # Add the first SimpleRNN Layer with unit count based on unit's input param
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))

    # Add the second SimpleRNN Layer with unit count based on unit's input param
    model.add(SimpleRNN(units[2]))

    # Add a Dropout layer of 0.2
    model.add(Dropout(0.2))

    # Add the output Layer with unit count based on unit's input param
    model.add(Dense(units[3], activation='sigmoid'))

    # Return the model from function scope
    return model
