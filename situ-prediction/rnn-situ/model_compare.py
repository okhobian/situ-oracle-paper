from data import *
from models import *
from history_callback import *
from plot import *

## VARIABLES
BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset_10_01_ignore_ts.csv'
ACTIVITIES = ['sleep', 'getup', 'eat', 'work', 'leisure', 'watchtv', 'rest', 'cook', 'goout', 'other']
WINDOW_SIZE = 15
LABEL_AHEAD = 1
MODELS = ['RNN', 'LSTM', 'GRU']

## LOAD DATA
data = DATASET()
data.load_data(DATA_FILE, ACTIVITIES)
histories = Histories()
plt = PLOT()

accuracies = []
losses = []
times = []
for m in MODELS:
    trainX, trainY = data.train_data(WINDOW_SIZE, LABEL_AHEAD)
    if m == 'RNN':
        model = build_rnn(trainX.shape[1], trainX.shape[2], trainY.shape[1])
    elif m == 'LSTM':
        model = build_lstm(trainX.shape[1], trainX.shape[2], trainY.shape[1])
    else:
        model = build_gru(trainX.shape[1], trainX.shape[2], trainY.shape[1])
        
    model.fit(trainX, trainY, batch_size=50, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),

    accuracies.append(histories.accuracies)
    losses.append(histories.losses)
    times.append(histories.times)

plt.add_multi_data_figure(accuracies, 'Accuracies', 'batch#', 'categorical_crossentropy', MODELS)
plt.add_multi_data_figure(losses, 'Losses', 'batch#', 'categorical_crossentropy', MODELS)
plt.add_multi_data_figure(times, 'Training time', 'batch#', 'time', MODELS)
plt.show_all()