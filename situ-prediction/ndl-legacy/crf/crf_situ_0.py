import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import sklearn_crfsuite
from sklearn_crfsuite import CRF, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 15

# Define the file path
file_path = '~/Desktop/research/situ-oracle-journal/dataset/openshs-classification/d1_1m_0tm.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# # Preparing data
# X = []
# y = []

# for i in range(0, len(df) - WINDOW_SIZE + 1):
#     # Convert all data to string format
#     X_window = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).iloc[i:i+WINDOW_SIZE].astype(str).values.tolist()
#     y_window_label = str(df['Activity'].iloc[i + WINDOW_SIZE - 1])
    
#     X.append(X_window)
#     y.append([y_window_label] * WINDOW_SIZE) # Replicate the final label for each item in the sequence

# # Splitting the dataset using TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=2)
# train_index, test_index = list(tscv.split(X))[0]

# X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
# y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

# # Training the CRF model
# crf = CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True
# )
# crf.fit(X_train, y_train)

# # Predicting on the test set
# y_pred = crf.predict(X_test)

# # Evaluate only the last label of each sequence for accuracy
# y_test_last = [seq[-1] for seq in y_test]
# y_pred_last = [seq[-1] for seq in y_pred]

# accuracy = accuracy_score(y_test_last, y_pred_last)
# print(f"Accuracy: {accuracy*100:.2f}%")


###################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# classes = ['sleep', 'eat', 'personal', 'work', 'leisure', 'other']

base_path = os.environ.get("OPENSHS_DATA_PATH")
if base_path:
    print("Base path:", base_path)
else:
    print("BASE_PATH environment variable not set.")

datasets = []
# for x in range(1, 2):
for x in [5]:
    datasets.append(base_path + f'd{x}_2m_0tm.csv')
    
    
for dataset in datasets:
    
    print(f"Training {dataset}")
    
    df = pd.read_csv(dataset)
    
    # Preparing data
    X = []
    y = []

    for i in range(0, len(df) - WINDOW_SIZE + 1):
        # Convert all data to string format
        X_window = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).iloc[i:i+WINDOW_SIZE].astype(str).values.tolist()
        # y_window_label = df['Activity'].iloc[i:i+WINDOW_SIZE].astype(str).values.tolist()
        # y_window_label1 = df['Activity'].iloc[i:i + WINDOW_SIZE].astype(str).values.tolist()
        y_window_label2 = [str(df['Activity'].iloc[i + WINDOW_SIZE - 1])] * WINDOW_SIZE
        # y_window_label = y_window_label1[:int(0.6 * WINDOW_SIZE)] + y_window_label2[:int(0.4 * WINDOW_SIZE)]
        
        X.append(X_window)
        y.append(y_window_label2)
        # y.append([y_window_label] * WINDOW_SIZE) # Replicate the final label for each item in the sequence

    # # Define the feature extraction function
    # def extract_features(sequence):
    #     features_sequence = []
    #     for i in range(len(sequence)):
    #         feature = {
    #             'current_value': str(sequence[i]),
    #             'is_first': i == 0,
    #             'is_last': i == len(sequence) - 1
    #         }
    #         if i > 0:
    #             feature['prev_value'] = str(sequence[i-1])
    #         if i < len(sequence) - 1:
    #             feature['next_value'] = str(sequence[i+1])
    #         # print(i, feature)
    #         features_sequence.append(feature)
    #     return features_sequence

    # # Transform X_sequences into feature format
    # X_features = [extract_features(seq) for seq in X]

    # Splitting the dataset using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)
    # train_index, test_index = list(tscv.split(X))[0]
    
    # Iterate through the TimeSeriesSplit to get indices
    train_indices = []
    test_indices = []

    for train_index, test_index in tscv.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    # Combine the first two train indices to form the training set
    combined_train_indices = train_indices[0].tolist() + train_indices[1].tolist()

    # The test indices from the second split form the test set
    final_test_indices = test_indices[1].tolist()
    

    X_train, X_test = [X[i] for i in combined_train_indices], [X[i] for i in final_test_indices]
    y_train, y_test = [y[i] for i in combined_train_indices], [y[i] for i in final_test_indices]

    # Record the start time
    start_time = time.time()

    # Training the CRF model
    crf = CRF(
        algorithm='lbfgs',
        # c1=0.1,
        # c2=0.1,
        # max_iterations=100,
        # all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Predicting on the test set
    y_pred = crf.predict(X_test)

    # Evaluate only the last label of each sequence for accuracy
    y_test_last = [seq[-1] for seq in y_test]
    y_pred_last = [seq[-1] for seq in y_pred]
    
    # print(y_test_last[100:130])
    # print("===")
    # print(y_pred_last[100:130])

    accuracy = accuracy_score(y_test_last, y_pred_last)
    # print(f"Accuracy: {accuracy*100:.2f}%")
    
    # results
    dataset_name = dataset.split('/')[-1]
    num_records = len(df)
    training_time = end_time - start_time
    
    print(f"{dataset_name} | {num_records} | {accuracy * 100:.2f}% | {training_time}")