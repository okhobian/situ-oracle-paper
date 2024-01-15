import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from web3 import Web3
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pickle
import hashlib
from keras.models import load_model
from os.path import dirname, join, abspath
import os

NUM_ROWS = 5000
context_history = np.zeros((15, 29))

MODEL_FILES = {
    "GNB": "situ_gnb.joblib",
    "BNB": "situ_bnb.joblib",
    "MNB": "situ_mnb.joblib",
    "RNN": "situ_rnn.joblib",
    "LSTM": "situ_lstm.joblib",
    "GRU": "situ_gru.joblib"
}

result_json = {
    "dataset": "",
    "train_size":0,
    "train_time":0, 
    "train_gas":0,
    "test_size":0,
    "test_accuracy":0,
    "avg_inference_time":0,
    "avg_inference_gas":0
}
training_times = []
training_gases = []
inference_times = []
inference_gases = []

sensor_list = ["wardrobe", "tv", "oven", "officeLight", "officeDoorLock", "officeDoor",	
  "officeCarp",	"office",	"mainDoorLock", "mainDoor",	"livingLight", "livingCarp",	
  "kitchenLight",	"kitchenDoorLock", "kitchenDoor", "kitchenCarp", "hallwayLight",	
  "fridge",	"couch", "bedroomLight",	"bedroomDoorLock", "bedroomDoor", "bedroomCarp",	
  "bedTableLamp",	"bed", "bathroomLight",	"bathroomDoorLock",	"bathroomDoor",	"bathroomCarp"
]

situ_list = ["sleep", "eat", "work", "leisure", "personal", "other"]

SENSOR_ACCOUNT = "0xb4DCbe9FE20CC56649e3Bc3D666FB62DB94c31d6"
SENSOR_PRIVATE_KEY = "a98bee319312213845bba60eb744081994adfc8b969598e3401096661d2dcc79"

ORACLE_ACCOUNT = "0x634c45E0926382307bBF316FB222f9bfbD1c251A"
ORACLE_PRIVATE_KEY = "e36f4b20774e01f907130d625dc6e531a4c0f8cd188efac355bb750963b4ff19"

# Get the current directory where the script is located
data_base_path = os.environ.get("OPENSHS_DATA_PATH")

current_directory = dirname(abspath(__file__))
dist_situ_build = join(current_directory, '../truffle/build/contracts/Situ.json')
dist_context_build = join(current_directory, '../truffle/build/contracts/Context.json')

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
print(web3.is_connected())

abi_situ = json.load(open(dist_situ_build))
cont_s = "0x5FD1E028E55055b09793E08dAeF2ca284d6C18ab"
cont_situ = web3.eth.contract(address=cont_s,abi=abi_situ['abi'])

abi_context = json.load(open(dist_context_build))
cont_c = "0x4f3EF3cf52E84f58226B99A7917179593c668c06"
cont_context = web3.eth.contract(address=cont_c,abi=abi_context['abi'])

# Function to update the context history with a new list of values
def update_context_history(history, new_values):
    history = np.roll(history, -1, axis=0)      # Roll the array 'up' and remove the first row
    history[-1, :] = new_values                 # Insert the new list of values at the end
    return history

def update_context_value(sensor:str, new_val:int, account, pkey):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_context.functions.update_context_value (
        sensor, new_val
    ).build_transaction({
        'gas': 600000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=pkey)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def update_situ(new_situ:str, oracle_account, pkey):
    nonce = web3.eth.get_transaction_count(oracle_account)
    tx = cont_situ.functions.update_situ (
        new_situ
    ).build_transaction({
        'gas': 600000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': oracle_account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=pkey)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

####################

datasets = []
for x in range(1,2):
# for x in [1,5,2,7,6]:
    datasets.append(data_base_path + f'd{x}_1m_0tm.csv')

for dataset in datasets:
    dataset_name = dataset.split('/')[-1]
    result_json['dataset'] = dataset_name   #====

    df = pd.read_csv(dataset, nrows = NUM_ROWS)
    X = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).values
    y = df['Activity'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0)
    result_json['test_size'] = len(X_test)      #====

    ################# UPDATE_CONTEXT_VALUE FROM SENSORS (Context.sol) -- one row at the time
    ## Load the RNN models from file
    model = load_model('gru.h5')
    
    gas_used_per_inference = []
    context_update_times = []
    model_inference_time = []
    for i in range(X_test.shape[0]):    # each row
        print(i)
        context_update_times.append(int(time.time()))   # record when context updated in epoch
        
        gas_per_row = 0
        row = X_test[i,:].tolist()
        for j, sensor_val in enumerate(row):
            sensor_str = sensor_list[j]
            tx_receipt = update_context_value(sensor_str, int(sensor_val), SENSOR_ACCOUNT, SENSOR_PRIVATE_KEY)
            gas_per_row += tx_receipt.gasUsed
        ################# 
        all_sensor_vals = []                ## GET_CONTEXT_VALUE FROM Context.sol -- (29 * 15) historical data
        for sensor_str in sensor_list:
            sensor_val = cont_context.functions.get_context_value(sensor_str).call() # 1 reading of this sensor
            all_sensor_vals.append(sensor_val) 
        
        # insert new readings to the cache
        context_history = update_context_history(context_history, all_sensor_vals) 
        
        # (15,29) input data
        input_data = context_history
        
        ## Load the bayes model from the file
        # model = load( MODEL_FILES['GNB'] )
        
        ## Make the prediction
        start_time = time.time()
        situ_str = situ_list[np.argmax(model.predict(input_data.reshape(1,15,29)), axis=-1)[0]]
        end_time = time.time()
        model_inference_time.append(end_time - start_time)
        print(situ_str)
        
        ## Update situ -- Situ.sol
        tx_receipt = update_situ( situ_str, ORACLE_ACCOUNT, ORACLE_PRIVATE_KEY )
        gas_per_row += tx_receipt.gasUsed            
        
        gas_used_per_inference.append(gas_per_row)
    
    # Save context_update_times to file
    df_time = pd.DataFrame(context_update_times, columns=['context_update_times'])
    df_gas = pd.DataFrame(gas_used_per_inference, columns=['gas_used_per_inference'])
    df_model_inf_time = pd.DataFrame(model_inference_time, columns=['model_inference_time'])
    df_time.to_csv('context_update_times.csv', index=False)
    df_gas.to_csv('gas_used_per_inference.csv', index=False)
    # df_model_inf_time.to_csv('df_model_inf_time', index=False)