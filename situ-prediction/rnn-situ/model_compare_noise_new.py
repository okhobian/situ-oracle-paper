import os
from data import *
from models import *
from history_callback import *
from plot import *

dataset_path = os.environ.get("OPENSHS_DATA_PATH")
if not dataset_path: print("OPENSHS_DATA_PATH environment variable not set.")

## VARIABLES
# DATA_FILE = dataset_path + f'd1_2m_0tm.csv'
ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'other']
WINDOW_SIZE = 15
LABEL_AHEAD = 1

result_ws_df = pd.DataFrame()
result_training_acc_df = pd.DataFrame()
result_training_los_df = pd.DataFrame()

train_accuracies = []
train_losses = []
train_times = []

train_time_per_ws = []
test_loss_per_ws = []
test_accuracy_per_ws = []

datasets = []
# for x in range(1, 8):
for x in [1]:
    datasets.append(dataset_path + f'd{x}_1m_0tm.csv')
    
try:
    for dataset in datasets:
        
        dataset_name = dataset.split('/')[-1]
        print(f"Training {dataset_name}")
        
        ## LOAD DATA
        data = DATASET(noise=True)
        data.load_data(dataset, ACTIVITIES)
        histories = Histories()
        plt = PLOT()

        train_data, test_data = data.split_data(test_percentage=0.4)

        # trainX, trainY, testX, testY = data.split_data(size, LABEL_AHEAD, test_percentage=0.3)
        trainX, trainY = data.form_data(train_data, WINDOW_SIZE, LABEL_AHEAD)
        testX,  testY  = data.form_data(test_data, WINDOW_SIZE, LABEL_AHEAD)
        
        model = build_rnn(trainX.shape[1], trainX.shape[2], trainY.shape[1])
        model.fit(trainX, trainY, batch_size=50, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),
        train_accuracies.append(histories.accuracies)
        train_losses.append(histories.losses)
        train_times.append(histories.times)
        
        test_loss, test_accuracy = model.evaluate(testX, testY)
        test_loss_per_ws.append(test_loss)
        test_accuracy_per_ws.append(test_accuracy)
        train_time_per_ws.append(sum(histories.times))

        result_ws_df[f"{dataset_name}"] = pd.Series([sum(histories.times), test_loss, test_accuracy])
        result_training_acc_df[f"{dataset_name}"] = pd.Series(histories.accuracies)
        result_training_los_df[f"{dataset_name}"] = pd.Series(histories.losses)

        print(f"{dataset_name} | TEST LOSS: {test_loss}")
        print(f"{dataset_name} | TEST ACCURACY: {test_accuracy}")
        print(f"{dataset_name} | TRAIN TIME: {sum(histories.times)}")
        
        # weights = []
        # for layer in model.layers:
        #     layer_weights = layer.get_weights()
        #     for w in layer_weights:
        #         weights.append(w.flatten())
        # weights = np.concatenate(weights)
        
        # print(weights)
        # print(weights.shape)
        
        # model.save('rnn_noise.h5')
        print("##############################################")

except Exception as e:
    pass

result_ws_df.to_csv('rnn_ws_n10.csv')
result_training_acc_df.to_csv('rnn_acc_n10.csv')
result_training_los_df.to_csv('rnn_los_n10.csv')





# plt.add_multi_data_figure(train_accuracies, 'Accuracies', 'batch#', 'accuracy', WINDOW_SIZE)
# plt.add_multi_data_figure(train_losses, 'Losses', 'batch#', 'loss', WINDOW_SIZE)
# plt.add_multi_data_figure(train_times, 'Training time', 'batch#', 'time', WINDOW_SIZE)
# plt.show_all()