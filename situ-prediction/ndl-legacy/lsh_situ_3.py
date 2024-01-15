import random
import numpy as np
from collections import defaultdict, Counter

## TimeSeriesSplit + raw acc

WINDOW_SIZE = 15

class LSHClassifier:
    def __init__(self, n_hash_tables=1000, n_hash_bits=8, window_size=WINDOW_SIZE):
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
            # prediction = random.choice(['sleep', 'eat', 'personal', 'work', 'leisure', 'other'])
            # prediction = None  # You can handle empty votes differently if needed
            prediction = 'personal'

        return prediction

###################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


def cross_entropy(y_true, y_pred):
    """
    Calculate cross-entropy for classification.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # clip the predicted probabilities to avoid log(0)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Define the classes
classes = ['sleep', 'eat', 'personal', 'work', 'leisure', 'other']


# Define the file path
file_path = '~/Desktop/research/situ-oracle-journal/dataset/openshs-classification/d1_2m_5tm.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Split the dataset into features (X) and labels (y)
X = df.drop(['Activity', 'wardrobe', 'timestamp'], axis=1).values
y = df['Activity'].values

# Initialize TimeSeriesSplit with the desired number of splits
tscv = TimeSeriesSplit(n_splits=2)

# Just to demonstrate, I'll use the first split. In a real scenario, you might want to use all splits for cross-validation.
train_index, test_index = next(tscv.split(X))

X_train_raw = X[train_index]
y_train_raw = y[train_index]
X_test_raw = X[test_index]
y_test_raw = y[test_index]

# Convert training and testing data to sequences of WINDOW_SIZE rows
X_train = np.array([X_train_raw[i:i+WINDOW_SIZE] for i in range(len(X_train_raw)-WINDOW_SIZE+1)])
y_train = y_train_raw[WINDOW_SIZE-1:]

X_test_sequences = np.array([X_test_raw[i:i+WINDOW_SIZE] for i in range(len(X_test_raw)-WINDOW_SIZE+1)])
y_test = y_test_raw[WINDOW_SIZE-1:]

clf = LSHClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
predictions = [clf.predict(X) for X in X_test_sequences]

# Calculate accuracy
correct_predictions = np.sum(predictions == y_test)
accuracy = correct_predictions / len(y_test)
print(f"Prediction Accuracy: {accuracy * 100:.2f}%")