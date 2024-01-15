import os
import time
import pandas as pd
import numpy as np
import random
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 15

base_path = os.environ.get("OPENSHS_DATA_PATH")
if base_path: print("Base path:", base_path)
else: print("BASE_PATH environment variable not set.")

datasets = []
# for x in range(1, 8):
for x in [1]:
    datasets.append(base_path + f'd{x}_1m_0tm.csv')
    
for dataset in datasets:
    
    print(f"Training {dataset}")
    df = pd.read_csv(dataset)

    X = []
    y = []

    for i in range(0, len(df) - WINDOW_SIZE + 1):   #df.iloc[:, :5]
        X_window = df.drop(columns=['Activity', 'timestamp']).iloc[i:i+WINDOW_SIZE].values.flatten().tolist()
        y_label = df['Activity'].iloc[i + WINDOW_SIZE - 1]
        
        X.append(X_window)
        y.append(y_label)
    
    # each row at once
    # X = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).values
    # y = df['Activity'].values

    # Splitting the dataset using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)
    train_index, test_index = list(tscv.split(X))[0]

    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

################## /GNB/BNB/MNB

    training_times = {}

    start_time = time.time()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    end_time = time.time()
    training_times['gnb'] = end_time - start_time
    
    start_time = time.time()
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    end_time = time.time()
    training_times['bnb'] = end_time - start_time
    
    start_time = time.time()
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    end_time = time.time()
    training_times['mnb'] = end_time - start_time
    
    # Predicting on the test set
    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)
    y_pred_mnb = mnb.predict(X_test)
    
    # get accuracy scores
    accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
    accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
    accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
    
    # results
    dataset_name = dataset.split('/')[-1]
    num_records = len(df)
    
    print(f"GNB | {dataset_name} | {num_records} | {accuracy_gnb * 100:.2f}% | {training_times['gnb']}")
    print(f"BNB | {dataset_name} | {num_records} | {accuracy_bnb * 100:.2f}% | {training_times['bnb']}")
    print(f"MNB | {dataset_name} | {num_records} | {accuracy_mnb * 100:.2f}% | {training_times['mnb']}")

    # Save the models to file
    dump(gnb, 'gnb.joblib')
    dump(bnb, 'bnb.joblib') 
    dump(mnb, 'mnb.joblib') 

