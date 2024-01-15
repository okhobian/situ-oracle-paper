from os.path import dirname, join, abspath
from os import environ
import time
import yaml
import json
import numpy as np
import pandas as pd

from web3.auto import w3
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
from web3.exceptions import TimeExhausted

from sklearn.model_selection import train_test_split
from joblib import load
from keras.models import load_model

#################### Get the current directory
curr_dir = dirname(abspath(__file__))

#################### Load config file
with open(join(curr_dir, join(curr_dir, 'config.yml')), 'r') as file:
    config = yaml.safe_load(file)

#################### Get the data & truffle project base path
data_base_path = environ.get("OPENSHS_DATA_PATH")
truffle_base_dir = environ.get("TRUFFLE_BASE_DIR")

#################### Blockchain parameters
network_url = config['chain']['network_url']
chain_id =  config['chain']['chain_id']

situ_contract_build = join(truffle_base_dir, config['chain']['situ_build'])
context_contract_build = join(truffle_base_dir, config['chain']['context_build'])
situ_contract_abi = json.load(open(situ_contract_build))['abi']
context_contract_abi = json.load(open(context_contract_build))['abi']
situ_contract_addr = json.load(open(situ_contract_build))['networks'][chain_id]['address']
context_contract_addr = json.load(open(context_contract_build))['networks'][chain_id]['address']

sensor_pbk = config['chain']['sensor_pbk']
sensor_pvk = config['chain']['sensor_pvk']

oracle_pbk = config['chain']['oracle_pbk']
oracle_pvk = config['chain']['oracle_pvk']

#################### Connect to the chain and init contract objects
web3 = Web3(HTTPProvider(network_url))
if config['chain']['is_poa']: web3.middleware_onion.inject(geth_poa_middleware, layer=0)    # specific for PoA
print("Network Connected: ", web3.is_connected())
situ_contract = web3.eth.contract(address=situ_contract_addr, abi=situ_contract_abi)
context_contract = web3.eth.contract(address=context_contract_addr, abi=context_contract_abi)

#################### Experimental parameters
    ## data
num_rows_from_dataset = config['data']['num_rows']
test_size = config['data']['test_size']
window_size = config['data']['window_size']
sensor_list = config['data']['sensor']
situ_list = config['data']['situ']
context_history = np.zeros((window_size, len(sensor_list)))
    ## model
model_name = config['model']['model_to_use']
is_model_rnn = config['model']['is_model_rnn']
model_to_eval = config['model']['model_files'][model_name]

#################### Results
# context_update_times_f = config['result']['context_update_times']
# actuation_times_f = config['result']['actuation_times']
# gas_used_per_inference_f = config['result']['gas_used_per_inference']
df_context_f= config['result']['df_context_tx']
df_situ_f= config['result']['df_situ_tx']

####################
def extract_receipt(tx_hashes):   # (time, hash)
    column_names = ['tx_hash', 'tx_sent_time', 'tx_mined_time', 'latency', 'gas_used', 'block_num', 'block_time', 'num_tx_in_block']
    tx_details = []
    total_num = len(tx_hashes)
    for i, (tx_time, tx_hash) in enumerate(tx_hashes):
        print(f">> {i}/{total_num} Extracting receipt for [{tx_hash}]")
        try:
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        except TimeExhausted:
            print(f"Transaction with hash {tx_hash} was not mined within the timeout period.")
                    
        block = web3.eth.get_block(receipt.blockNumber)                 # Get the block
        previous_block = web3.eth.get_block(receipt.blockNumber - 1)    # Get the previous block
        block_time = block.timestamp - previous_block.timestamp         # Calculate the block time
        row_data = {
            'tx_hash': tx_hash, 
            'tx_sent_time': tx_time,
            'tx_mined_time': block.timestamp,
            'latency': block.timestamp - tx_time,
            'gas_used': receipt.gasUsed,
            'block_num': receipt.blockNumber, 
            'block_time': block_time, 
            'num_tx_in_block': len(block.transactions)
        }
        tx_details.append(row_data)
        
    return pd.DataFrame(tx_details, columns=column_names)
    
# Function to update the context history with a new list of values
def update_context_history(history, new_values):
    history = np.roll(history, -1, axis=0)      # Roll the array 'up' and remove the first row
    history[-1, :] = new_values                 # Insert the new list of values at the end
    return history

def update_context_value(nonce:int, sensor:str, new_val:int, account, pkey):
    # nonce = web3.eth.get_transaction_count(account)
    # print(nonce)
    tx = context_contract.functions.update_context_value (
        sensor, new_val
    ).build_transaction({
        # 'gas': 6000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=pkey)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] update_context_value: {int(time.time())} | {sensor} | {new_val} | {tx_hash.hex()}")
    return tx_hash
    # tx_queue.append(tx_hash) # enqueue trasaction hash
    # tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    # block = web3.eth.get_block(tx_receipt.blockNumber)                 # Get the block
    # previous_block = web3.eth.get_block(tx_receipt.blockNumber - 1)    # Get the previous block
    # block_time = block.timestamp - previous_block.timestamp         # Calculate the block time
    # num_transactions = len(block.transactions)  # The number of transactions in the block is the length of the transactions array
    # return tx_receipt

def update_situ(nonce:int, new_situ:str, oracle_account, pkey):
    # nonce = web3.eth.get_transaction_count(oracle_account)
    tx = situ_contract.functions.update_situ (
        new_situ
    ).build_transaction({
        # 'gas': 6000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': oracle_account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=pkey)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] update_situ: {int(time.time())} | {new_situ} | {tx_hash.hex()}")
    return tx_hash
    
    # tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    # block = web3.eth.get_block(tx_receipt.blockNumber)                 # Get the block
    # previous_block = web3.eth.get_block(tx_receipt.blockNumber - 1)    # Get the previous block
    # block_time = block.timestamp - previous_block.timestamp         # Calculate the block time
    # num_transactions = len(block.transactions)  # The number of transactions in the block is the length of the transactions array
    # print(f"[MINED] update_situ: {int(time.time())} | {new_situ}| {tx_receipt.gasUsed} | {tx_receipt.blockNumber} | {block_time} | {num_transactions}")
    # return tx_receipt

####################

dataset = join(data_base_path, config['data']['dataset'])
df = pd.read_csv(dataset, nrows = num_rows_from_dataset)
X = df.drop(columns=['Activity', 'timestamp']).values
y = df['Activity'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=0)

model = None
if is_model_rnn: 
    model = load_model(model_to_eval)   ## Load the RNN models from file
else:
    model = load( model_to_eval )    ## Load the bayes model from the file (joblib)

################# UPDATE_CONTEXT_VALUE FROM SENSORS (call Context.sol) -- one row at the time
# gas_used_per_inference = []
# context_update_times = []

context_tx_hashes = []
situ_tx_hashes = []
sensor_nonce = web3.eth.get_transaction_count(sensor_pbk, 'pending')
oracle_nonce = web3.eth.get_transaction_count(oracle_pbk, 'pending')

for i in range(X_test.shape[0]):    # each row
    # sensor_nonce = web3.eth.get_transaction_count(sensor_pbk, 'pending')
    # oracle_nonce = web3.eth.get_transaction_count(oracle_pbk, 'pending')
    print("sensor_pbk nonce", sensor_nonce)
    print("oracle_pbk nonce", oracle_nonce)
    

    # context_update_times.append(int(time.time()))   # record when context updated in epoch
    
    # gas_per_row = 0
    row = X_test[i,:].tolist()
    for j, sensor_val in enumerate(row):
        sensor_str = sensor_list[j]
        # tx_receipt = update_context_value(sensor_str, int(sensor_val), sensor_pbk, sensor_pvk)  ## UPDATE_CONTEXT_VALUE
        
        tx_hash = update_context_value(sensor_nonce, sensor_str, int(sensor_val), sensor_pbk, sensor_pvk)  ## UPDATE_CONTEXT_VALUE
        context_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        sensor_nonce += 1
        # gas_per_row += tx_receipt.gasUsed
    ################# 
    
    all_sensor_vals = []                ## GET_CONTEXT_VALUE FROM Context.sol -- (29 * 15) historical data
    for sensor_str in sensor_list:
        sensor_val = context_contract.functions.get_context_value(sensor_str).call() # 1 reading of this sensor
        all_sensor_vals.append(sensor_val) 
    
    # insert new readings to the cache
    context_history = update_context_history(context_history, all_sensor_vals)
    
    ## Make the prediction
    input_data = context_history    # (15,29) input data
    
    if is_model_rnn:    # Keras rnn models
        situ_str = situ_list[np.argmax(model.predict(input_data.reshape(1,window_size,len(sensor_list))), axis=-1)[0]]
    else:               # Bayes models
        input_data = input_data.flatten().reshape(1, -1)
        situ_str = model.predict(input_data)[0]
    
    ## Update situ -- call Situ.sol
    
    try:
        tx_hash = update_situ(oracle_nonce, situ_str, oracle_pbk, oracle_pvk )
        situ_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        oracle_nonce += 1
        time.sleep(3)
    except:
        oracle_nonce += 1
        
    
    # gas_per_row += tx_receipt.gasUsed            
    # gas_used_per_inference.append(gas_per_row)
    
    print(f"{i}: {situ_str}")

# Save results to files
# df_time = pd.DataFrame(context_update_times, columns=['context_update_times'])
# df_gas = pd.DataFrame(gas_used_per_inference, columns=['gas_used_per_inference'])
# df_time.to_csv(join(curr_dir, context_update_times_f), index=False)
# df_gas.to_csv(join(curr_dir, gas_used_per_inference_f), index=False)

df_context = extract_receipt(context_tx_hashes)
df_situ = extract_receipt(situ_tx_hashes)

df_context.to_csv(join(curr_dir, df_context_f), index=False)
df_situ.to_csv(join(curr_dir, df_situ_f), index=False)

print(df_context)
print("---")
print(df_situ)