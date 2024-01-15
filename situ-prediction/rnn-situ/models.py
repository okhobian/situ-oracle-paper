from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, RNN, SimpleRNN


def build_lstm(trainX_window_size, trainX_feature_length, trainY_num_categories):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX_window_size, trainX_feature_length), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY_num_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def build_rnn(trainX_window_size, trainX_feature_length, trainY_num_categories):
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', input_shape=(trainX_window_size, trainX_feature_length), return_sequences=False))
    # model.add(RNN(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY_num_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def build_gru(trainX_window_size, trainX_feature_length, trainY_num_categories):
    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=(trainX_window_size, trainX_feature_length), return_sequences=True))
    model.add(GRU(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY_num_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model