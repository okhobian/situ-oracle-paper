from data import *
from models import *
from history_callback import *
from plot import *
from tqdm import tqdm

## VARIABLES
BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset_10_01_ignore_ts.csv'
# ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']
ACTIVITIES = ['sleep', 'getup', 'eat', 'work', 'leisure', 'watchtv', 'rest', 'cook', 'goout', 'other']
# WINDOW_SIZE = [5, 10]
WINDOW_SIZE = [5,7,9,11,13,15,17,19,21,23,25,27,29]
# WINDOW_SIZE = [5]
LABEL_AHEAD = 1

result_ws_df = pd.DataFrame()
result_training_acc_df = pd.DataFrame()
result_training_los_df = pd.DataFrame()

## LOAD DATA
data = DATASET(noise=True)
data.load_data(DATA_FILE, ACTIVITIES)
histories = Histories()
plt = PLOT()

# train_data, test_data = data.split_data(test_percentage=0.3)
train_data = pd.read_csv('train_noise.csv')
test_data = pd.read_csv('test_noise.csv')


train_accuracies = []
train_losses = []
train_times = []

train_time_per_ws = []
test_loss_per_ws = []
test_accuracy_per_ws = []

for size in WINDOW_SIZE:
    print(f"model [GRU] @ ws={size}")
    
    trainX, trainY = data.form_data(train_data, size, LABEL_AHEAD)
    testX,  testY  = data.form_data(test_data, size, LABEL_AHEAD)
    
    model = build_gru(trainX.shape[1], trainX.shape[2], trainY.shape[1])
    model.fit(trainX, trainY, batch_size=50, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),
    train_accuracies.append(histories.accuracies)
    train_losses.append(histories.losses)
    train_times.append(histories.times)
    
    test_loss, test_accuracy = model.evaluate(testX, testY)
    test_loss_per_ws.append(test_loss)
    test_accuracy_per_ws.append(test_accuracy)
    train_time_per_ws.append(sum(histories.times))

    result_ws_df[f"T={str(size)}"] = pd.Series([sum(histories.times), test_loss, test_accuracy])
    result_training_acc_df[f"T={str(size)}"] = pd.Series(histories.accuracies)
    result_training_los_df[f"T={str(size)}"] = pd.Series(histories.losses)

    print(f"WS={size} | TEST LOSS: {test_loss}")
    print(f"WS={size} | TEST ACCURACY: {test_accuracy}")
    print(f"WS={size} | TRAIN TIME: {sum(histories.times)}")

result_ws_df.to_csv('gru_ws.csv')
result_training_acc_df.to_csv('gru_acc.csv')
result_training_los_df.to_csv('gru_los.csv')


# plt.add_multi_data_figure(train_accuracies, 'Accuracies', 'batch#', 'accuracy', WINDOW_SIZE)
# plt.add_multi_data_figure(train_losses, 'Losses', 'batch#', 'loss', WINDOW_SIZE)
# plt.add_multi_data_figure(train_times, 'Training time', 'batch#', 'time', WINDOW_SIZE)
# plt.show_all()