import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 15

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

    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # Record the start time
    start_time = time.time()

    # Training the Decision Tree model
    clf = DecisionTreeClassifier(criterion="entropy", splitter="random", min_samples_split=3, max_leaf_nodes=6, max_depth=2)
    clf.fit(X_train, y_train)
    
    # Record the end time
    end_time = time.time()

    # Predicting on the test set
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # results
    dataset_name = dataset.split('/')[-1]
    num_records = len(df)
    training_time = end_time - start_time
    
    print(f"{dataset_name} | {num_records} | {accuracy * 100:.2f}% | {training_time}")