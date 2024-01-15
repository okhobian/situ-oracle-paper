from data import *
from models import *
from history_callback import *
from plot import *

## VARIABLES
BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset_10_01_ignore_ts.csv'
# ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']
ACTIVITIES = ['sleep', 'getup', 'eat', 'work', 'leisure', 'watchtv', 'rest', 'cook', 'goout', 'other']

WINDOW_SIZE = 15
LABEL_AHEAD = 1

## LOAD DATA
data = DATASET()
data.load_data(DATA_FILE, ACTIVITIES)
trainX, trainY = data.train_data(WINDOW_SIZE, LABEL_AHEAD)

## BUILD MODEL
model = build_lstm(trainX.shape[1], trainX.shape[2], trainY.shape[1])

## TRAIN MODEL
histories = Histories()
model.fit(trainX, trainY, batch_size=20, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),

## RESULTS
plt = PLOT()
plt.add_figure(histories.accuracies, 'model accuracies', 'batch', 'accuracy', ['accuracy'])
plt.add_figure(histories.losses, 'model losses', 'batch', 'loss', ['loss'])
plt.add_figure(histories.times, 'training time', 'batch', 'time', ['time'])
plt.show_all()