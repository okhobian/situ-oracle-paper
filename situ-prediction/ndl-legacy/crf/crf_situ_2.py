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
file_path = '~/Desktop/research/situ-oracle-journal/dataset/openshs-classification/d1_2m_0tm.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Assume df is loaded as your DataFrame
columns_to_drop = ['Activity', 'wardrobe', 'timestamp']
X = df.drop(columns=columns_to_drop).values
y = df['Activity'].values

# Extract sequences for X and y
X_sequences = [X[i:i+WINDOW_SIZE] for i in range(0, len(X) - WINDOW_SIZE)]
y_sequences = [y[i:i+WINDOW_SIZE] for i in range(0, len(y) - WINDOW_SIZE)]

# Define the feature extraction function
def extract_features(sequence):
    features_sequence = []
    for i in range(len(sequence)):
        feature = {
            'current_value': str(sequence[i]),
            'is_first': i == 0,
            'is_last': i == len(sequence) - 1
        }
        if i > 0:
            feature['prev_value'] = str(sequence[i-1])
        if i < len(sequence) - 1:
            feature['next_value'] = str(sequence[i+1])
        features_sequence.append(feature)
    return features_sequence

# Transform X_sequences into feature format
X_features = [extract_features(seq) for seq in X_sequences]

# Split data
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=2)
train_index, test_index = next(tscv.split(X_features))

X_train, X_test = [X_features[i] for i in train_index], [X_features[i] for i in test_index]
y_train, y_test = [y_sequences[i] for i in train_index], [y_sequences[i] for i in test_index]

# Train a CRF model
crf = CRF(
    algorithm='lbfgs',
    max_iterations=10,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predict on test set
y_pred = crf.predict(X_test)

# Evaluate the results
# print(flat_classification_report(y_test, y_pred))

# correct_predictions = 0
# total_predictions = 0

# for true_seq, pred_seq in zip(y_test, y_pred):
#     for true_label, pred_label in zip(true_seq, pred_seq):
#         if true_label == pred_label:
#             correct_predictions += 1
#         total_predictions += 1

# accuracy = (correct_predictions / total_predictions) * 100
# print(f"Accuracy: {accuracy:.2f}%")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# results
# dataset_name = dataset.split('/')[-1]
# num_records = len(df)
# training_time = end_time - start_time

# print(f"{dataset_name} | {num_records} | {accuracy * 100:.2f}% | {training_time}")