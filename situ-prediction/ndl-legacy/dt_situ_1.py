import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 15

# Define the file path
file_path = '~/Desktop/research/situ-oracle-journal/dataset/openshs-classification/d1_2m_0tm.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Preparing data
X = []
y = []

for i in range(0, len(df) - WINDOW_SIZE + 1):
    X_window = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).iloc[i:i+WINDOW_SIZE].values.flatten().tolist()
    y_label = df['Activity'].iloc[i + WINDOW_SIZE - 1]
    
    X.append(X_window)
    y.append(y_label)

# Splitting the dataset using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(X))[0]

X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

# Training the Decision Tree model
clf = DecisionTreeClassifier(criterion="entropy", splitter="random", min_samples_split=3, max_leaf_nodes=6, max_depth=2)
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")