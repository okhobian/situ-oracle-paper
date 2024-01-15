import numpy as np
from os.path import dirname, join, abspath
import pickle
import hashlib

# Function to calculate SHA-256 hash of a file
def calculate_file_hash(filename, hash_function='sha256'):
    # Choose the hashing algorithm
    h = hashlib.new(hash_function)
    
    # Open the file in binary mode
    with open(filename, 'rb') as file:
        # Read and update hash in chunks of 4K
        for chunk in iter(lambda: file.read(4096), b""):
            h.update(chunk)
    # Return the hexadecimal digest of the hash
    return h.hexdigest()

curr_dir = dirname(abspath(__file__))
model_hash = calculate_file_hash(join(curr_dir, 'mnb.joblib'))
print(model_hash)