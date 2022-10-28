"""
Definition of NN model
"""
import keras
from keras import regularizers, Input
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU, SimpleRNN
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


def get_sae(x_train, hidden, output):
    """SAE(Auto-Encoders) TODO REPLACE THIS
    Build SAE Model.
    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    input_dim = x_train[0].shape[0]
    # learning_rate = 1e-5
    learning_rate = 1e-5
    model = Sequential()
    model.add(Flatten())
    # first layer
    model.add(Dense(hidden, input_dim=input_dim, name='input', activation="relu"))

    # second layer
    model.add(Dense(hidden / 2, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Third layer
    #model.add(Dense(hidden / 4, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # add dropout
    # model.add(Dropout(0.2))

    # Fourth/Bottleneck layer
    model.add(Dense(hidden / 4, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Fifth layer
    #model.add(Dense(hidden / 4, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    model.add(Dropout(0.2))

    # Sixth layer
    model.add(Dense(hidden / 2, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # add dropout
    model.add(Dropout(0.2))

    model.add(Dense(hidden, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))

    # Output layer
    model.add(Dense(output, activation="sigmoid"))
    model.add(keras.layers.Reshape((-1,96)))

    return model


def get_sae2(x_train, output):
    batch_size = 32
    input_dim = x_train[0].shape[0]  # num of predictor variables
    learning_rate = 3e-5
    # Input Layer
    input_layer = Input(shape=(input_dim,), name='input')
    # Encoder’s first dense layer
    encoder = Dense(2000, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dropout(0.3)(encoder)
    # Encoder’s second dense layer
    encoder = Dense(1000, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(encoder)

    encoder = Dropout(0.3)(encoder)
    # Encoder’s third dense layer
    encoder = Dense(500, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(encoder)
    encoder = Dropout(0.3)(encoder)
    # Code layer
    encoder = Dense(200, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(encoder)
    # Decoder’s first dense layer
    decoder = Dense(500, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(encoder)
    # Decoder’s second dense layer
    # add dropout
    decoder = Dropout(0.3)(decoder)

    decoder = Dense(1000, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(decoder)
    decoder = Dropout(0.3)(decoder)
    # Decoder’s Third dense layer
    decoder = Dense(2000, activation='relu',activity_regularizer=regularizers.l1(learning_rate))(decoder)
    decoder = Dropout(0.3)(decoder)
    # Output Layer
    decoder = Dense(output, activation='sigmoid', activity_regularizer=regularizers.l1(learning_rate))(decoder)

    model = keras.Model(inputs=input_layer, outputs=decoder)

    return model

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
