import numpy as np
import random
import os
import pickle
import hashlib

# Function to update the context history with a new list of values
def update_context_history(history, new_values):
    # Roll the array 'up' and remove the first row
    history = np.roll(history, -1, axis=0)
    # Insert the new list of values at the end
    history[-1, :] = new_values
    return history

sensor_list = ["wardrobe", "tv", "oven", "officeLight", "officeDoorLock",	"officeDoor",	
  "officeCarp",	"office",	"mainDoorLock", "mainDoor",	"livingLight", "livingCarp",	
  "kitchenLight",	"kitchenDoorLock", "kitchenDoor", "kitchenCarp", "hallwayLight",	
  "fridge",	"couch", "bedroomLight",	"bedroomDoorLock", "bedroomDoor", "bedroomCarp",	
  "bedTableLamp",	"bed", "bathroomLight",	"bathroomDoorLock",	"bathroomDoor",	"bathroomCarp"
];

# context_history = np.zeros((15, 29))

# for i in range(16):
#     all_sensor_vals = []                ## GET_CONTEXT_VALUE FROM Context.sol -- (29 * 15) historical data
#     for sensor_str in sensor_list:
#         sensor_val = random.randint(0,1) # 1 reading of this sensor
#         all_sensor_vals.append(sensor_val) 
#     context_history = update_context_history(context_history, all_sensor_vals)
# print(context_history)
# print(context_history.shape)


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

from keras.models import load_model
model = load_model('rnn.h5')

#  ## Serialize the model to a byte stream
# model_bytes = pickle.dumps(model)

# ## Compute the SHA-256 hash of the serialized model
# model_hash = hashlib.sha256(model_bytes).hexdigest()
model_hash = calculate_file_hash('gru.h5')

## Print the SHA-256 hash
print(model_hash)

situ_list = ["sleep", "eat", "work", "leisure", "personal", "other"]

# from data import *
# data = DATASET()
# data_base_path = os.environ.get("OPENSHS_DATA_PATH")
# data.load_data(data_base_path + f'd{1}_1m_0tm.csv', situ_list)
# train_data, test_data = data.split_data(test_percentage=0.4)
# testX,  testY  = data.form_data(test_data, 15, 1)

# print(testX.shape)

# situ = situ_list[np.argmax(model.predict(context_history.reshape(1,15,29)), axis=-1)[0]]
# print(situ)