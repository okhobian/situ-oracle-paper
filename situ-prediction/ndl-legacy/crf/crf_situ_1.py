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

# Preparing data
X = []
y = []

for i in range(0, len(df) - WINDOW_SIZE + 1):
    # Convert all data to string format
    X_window = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).iloc[i:i+WINDOW_SIZE].astype(str).values.tolist()
    y_window_label = str(df['Activity'].iloc[i + WINDOW_SIZE - 1])
    
    X.append(X_window)
    y.append([y_window_label] * WINDOW_SIZE) # Replicate the final label for each item in the sequence

# Splitting the dataset using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=2)
train_index, test_index = list(tscv.split(X))[0]

X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

# Training the CRF model
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predicting on the test set
y_pred = crf.predict(X_test)

# Evaluate only the last label of each sequence for accuracy
y_test_last = [seq[-1] for seq in y_test]
y_pred_last = [seq[-1] for seq in y_pred]

accuracy = accuracy_score(y_test_last, y_pred_last)
print(f"Accuracy: {accuracy*100:.2f}%")