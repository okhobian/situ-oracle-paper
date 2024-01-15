import os
import time
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import TimeSeriesSplit

WINDOW_SIZE = 15

class LSH:
    def __init__(self, num_tables, hash_size, vector_length):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables = []
        
        # Randomly generated hash functions for each table
        for _ in range(num_tables):
            self.tables.append({})

    def _hash(self, vector):
        # Return a list of hashes for the vector, one per table
        return [self._hash_for_table(vector, table) for table in range(self.num_tables)]
    
    def _hash_for_table(self, vector, table):
        random.seed(table)
        hash_code = ''
        for _ in range(self.hash_size):
            idx = random.randint(0, len(vector) - 1)
            hash_code += str(vector[idx])
        return hash_code
    
    def insert(self, vector, label):
        # Insert the vector and its label into the hash tables
        hashes = self._hash(vector)
        for table, hash_code in enumerate(hashes):
            if hash_code not in self.tables[table]:
                self.tables[table][hash_code] = []
            self.tables[table][hash_code].append(label)
    
    def predict(self, vector):
        # Predict label based on approximate nearest neighbors
        hashes = self._hash(vector)
        labels = []
        for table, hash_code in enumerate(hashes):
            if hash_code in self.tables[table]:
                labels.extend(self.tables[table][hash_code])
        return max(set(labels), key=labels.count) if labels else None

# Example usage:
base_path = os.environ.get("OPENSHS_DATA_PATH")
if base_path:
    print("Base path:", base_path)
else:
    print("BASE_PATH environment variable not set.")

datasets = []
# for x in range(1, 8):
for x in [1,5,2,7,6]:
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

    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # Record the start time
    start_time = time.time()

    lsh = LSH(num_tables=5, hash_size=10, vector_length=WINDOW_SIZE*29)

    for X, y in zip(X_train, y_train):
        lsh.insert(X, y)

    # Record the end time
    end_time = time.time()

    # Predict on the test set
    predictions = [lsh.predict(X) for X in X_test]

    # Calculate accuracy
    predictions = np.array(predictions)
    y_test = np.array(y_test)
    correct_predictions = np.sum(predictions == y_test)
    print(correct_predictions)
    accuracy = correct_predictions / len(y_test)

    # results
    dataset_name = dataset.split('/')[-1]
    num_records = len(df)
    training_time = end_time - start_time
    
    print(f"{dataset_name} | {num_records} | {accuracy * 100:.2f}% | {training_time}")
