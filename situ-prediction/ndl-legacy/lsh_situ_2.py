import random
import numpy as np
from collections import defaultdict, Counter

## train_test_split + ce

WINDOW_SIZE = 15

class LSHClassifier:
    def __init__(self, n_hash_tables=10, n_hash_bits=8, window_size=WINDOW_SIZE):
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

# Split the dataset into training (60%) and testing (40%) sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.4, random_state=40)

# Convert training and testing data to sequences of WINDOW_SIZE rows
X_train = np.array([X_train_raw[i:i+WINDOW_SIZE] for i in range(len(X_train_raw)-WINDOW_SIZE+1)])
y_train = y_train_raw[WINDOW_SIZE-1:]

X_test_sequences = np.array([X_test_raw[i:i+WINDOW_SIZE] for i in range(len(X_test_raw)-WINDOW_SIZE+1)])
y_test = y_test_raw[WINDOW_SIZE-1:]

clf = LSHClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
predictions = [clf.predict(X) for X in X_test_sequences]

# Convert y_test and predictions to one-hot encoding for cross-entropy calculation
y_true_onehot = np.eye(len(classes))[np.array([classes.index(label) for label in y_test])]
predictions_onehot = np.eye(len(classes))[np.array([classes.index(label) for label in predictions])]

# Equal probability for each class if random prediction
for i, label in enumerate(predictions):
    if label == "random_guess":
        predictions_onehot[i] = [1/len(classes)] * len(classes)

# # Calculate cross entropy
ce = cross_entropy(y_true_onehot, predictions_onehot)
print(f"Cross-Entropy: {ce:.2f}")