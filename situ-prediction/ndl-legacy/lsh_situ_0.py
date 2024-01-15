import os
import random
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

## TimeSeriesSplit + raw acc

WINDOW_SIZE = 15

class LSHClassifier():
    def __init__(self, n_hash_tables=30, n_hash_bits=8, window_size=WINDOW_SIZE):
        self.n_hash_tables = n_hash_tables
        self.n_hash_bits = n_hash_bits
        self.window_size = window_size
        self.hash_tables = []
        self.rng = np.random.default_rng()

    def _generate_random_hyperplanes(self, n_features, n_hyperplanes):
        return self.rng.normal(size=(n_hyperplanes, n_features))

    def fit(self, X, y):
        n_features = X.shape[2] * self.window_size
        self.hash_tables = []

        # Flatten the WINDOW_SIZE-row window into a single row
        X_flattened = X.reshape(X.shape[0], -1)

        for _ in range(self.n_hash_tables):
            hyperplanes = self._generate_random_hyperplanes(n_features, self.n_hash_bits)
            hash_table = defaultdict(list)

            for xi, yi in zip(X_flattened, y):
                hash_code = (np.dot(xi, hyperplanes.T) > 0).astype(int)
                hash_code_str = ''.join(map(str, hash_code))
                hash_table[hash_code_str].append(yi)

            self.hash_tables.append(hash_table)

    def predict(self, X):
        # Flatten the WINDOW_SIZE-row window into a single row
        xi = X.reshape(-1)

        votes = []

        for table in self.hash_tables:
            hyperplanes = self._generate_random_hyperplanes(xi.shape[0], self.n_hash_bits)
            hash_code = (np.dot(xi, hyperplanes.T) > 0).astype(int)
            hash_code_str = ''.join(map(str, hash_code))

            if hash_code_str in table:
                most_common_label = Counter(table[hash_code_str]).most_common(1)[0][0]
                votes.append(most_common_label)

        if votes:
            prediction = Counter(votes).most_common(1)[0][0]
        else:
            # prediction = Counter(votes).most_common(1)[1][0]
            prediction = random.choice(['sleep', 'eat', 'personal', 'work', 'leisure', 'other'])
            # prediction = None  # You can handle empty votes differently if needed
            # prediction = 'personal'

        return prediction
    

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
# for x in range(1, 8):
for x in [1,2,6]:
    datasets.append(base_path + f'd{x}_2m_0tm.csv')
    
    
for dataset in datasets:
    
    print(f"Training {dataset}")
    
    df = pd.read_csv(dataset)
    
    # Preparing data
    X = []
    y = []

    for i in range(0, len(df) - WINDOW_SIZE + 1):
        X_window = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).iloc[i:i+WINDOW_SIZE].values.flatten().tolist()
        y_label = df['Activity'].iloc[i + WINDOW_SIZE - 1]
        
        X.append(X_window)
        y.append(y_label)

    # Splitting the dataset using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)
    train_index, test_index = list(tscv.split(X))[0]

    X_train, X_test = np.array([X[i] for i in train_index], [X[i] for i in test_index])
    y_train, y_test = np.array([y[i] for i in train_index], [y[i] for i in test_index])
    
    # # Split the dataset into features (X) and labels (y)
    # X = df.drop(['Activity', 'wardrobe', 'timestamp'], axis=1).values
    # y = df['Activity'].values
    
    # # Split the dataset into training (60%) and testing (40%) sets
    # X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.4, random_state=40)
    
    # # Convert training and testing data to sequences of WINDOW_SIZE rows
    # X_train = np.array([X_train_raw[i:i+WINDOW_SIZE] for i in range(len(X_train_raw)-WINDOW_SIZE+1)])
    # y_train = y_train_raw[WINDOW_SIZE-1:]   # first label at row 15, then shift by 1 for every next training sample

    # X_test = np.array([X_test_raw[i:i+WINDOW_SIZE] for i in range(len(X_test_raw)-WINDOW_SIZE+1)])
    # y_test = y_test_raw[WINDOW_SIZE-1:]
    
    # # Initialize TimeSeriesSplit with the desired number of splits
    # tscv = TimeSeriesSplit(n_splits=2)
    # train_index, test_index = next(tscv.split(X))
    # X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    # y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # # Convert training and testing data to sequences of WINDOW_SIZE rows
    # X_train = np.array([X_train[i:i+WINDOW_SIZE] for i in range(len(X_train)-WINDOW_SIZE+1)])
    # y_train = y_train[WINDOW_SIZE-1:]   # first label at row 15, then shift by 1 for every next training sample

    # X_test = np.array([X_test[i:i+WINDOW_SIZE] for i in range(len(X_test)-WINDOW_SIZE+1)])
    # y_test = y_test[WINDOW_SIZE-1:]
    
    # Record the start time
    start_time = time.time()
    
    # train model
    clf = LSHClassifier()
    clf.fit(X_train, y_train)
    
    # Record the end time
    end_time = time.time()
    
    # Predict on the test set
    predictions = [clf.predict(X) for X in X_test]

    # Calculate accuracy
    correct_predictions = np.sum(predictions == y_test)
    accuracy = correct_predictions / len(y_test)
    # print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # results
    dataset_name = dataset.split('/')[-1]
    num_records = len(df)
    training_time = end_time - start_time
    
    print(f"{dataset_name} | {num_records} | {accuracy * 100:.2f}% | {training_time}")