import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import time
import yaml

from web3.auto import w3
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
from web3.exceptions import TimeExhausted

from os.path import dirname, join, abspath
from os import environ

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

train_contract_build = join(truffle_base_dir, config['chain']['bayes_train_build'])
predict_contract_build = join(truffle_base_dir, config['chain']['bayes_predict_build'])
train_contract_abi = json.load(open(train_contract_build))['abi']
predict_contract_abi = json.load(open(predict_contract_build))['abi']
train_contract_addr = json.load(open(train_contract_build))['networks'][chain_id]['address']
predict_contract_addr = json.load(open(predict_contract_build))['networks'][chain_id]['address']

sensor_pbk = config['chain']['sensor_pbk']
sensor_pvk = config['chain']['sensor_pvk']

#################### Connect to the chain and init contract objects
web3 = Web3(HTTPProvider(network_url))
if config['chain']['is_poa']: web3.middleware_onion.inject(geth_poa_middleware, layer=0)    # specific for PoA
print("Network Connected: ", web3.is_connected())
train_contract = web3.eth.contract(address=train_contract_addr, abi=train_contract_abi)
predict_contract = web3.eth.contract(address=predict_contract_addr, abi=predict_contract_abi)

#################### Experimental parameters
    ## data
num_rows_from_dataset = config['data']['num_rows']
test_size = config['data']['test_size']
window_size = config['data']['window_size']
sensor_list = config['data']['sensor']
situ_list = config['data']['situ']
context_history = np.zeros((window_size, len(sensor_list)))

#################### Results
df_train_f= config['result']['df_train_tx']
df_predict_f= config['result']['df_predict_tx']

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
situ_int_mapping = {'sleep':0, 'eat':1, 'other':2, 'work':3, 'leisure':4, 'personal':5}

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


def cal_mean(nonce:int, feature, n, account, private_key, contract):
    # nonce = web3.eth.get_transaction_count(account)
    tx = contract.functions.cal_mean(
        feature,n
    ).build_transaction({
        'gas': 5000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] cal_mean: {int(time.time())} | {tx_hash.hex()}")
    return tx_hash
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def cal_variance(nonce:int, feature, mean, n, account, private_key, contract):
    # nonce = web3.eth.get_transaction_count(account)
    tx = contract.functions.cal_variance(
        feature,mean,n
    ).build_transaction({
        'gas': 5000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] cal_variance: {int(time.time())} | {tx_hash.hex()}")
    return tx_hash
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt
    
def cal_class(nonce:int, features, variances, means, account, private_key, contract):
    # nonce = web3.eth.get_transaction_count(account)
    tx = contract.functions.cal_parameters(
        features ,variances,means 
    ).build_transaction({
        'gas': 29999999,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] cal_parameters: {int(time.time())} | {tx_hash.hex()}")
    return tx_hash
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def cal_probability(nonce:int, Prior0, Prior1, Prior2, Prior3, Prior4, Prior5,account, private_key, contract):
    # nonce = web3.eth.get_transaction_count(account)
    tx = contract.functions.cal_probability (
        Prior0, Prior1, Prior2, Prior3, Prior4, Prior5
    ).build_transaction({
        'gas': 29999999,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"[SENT] cal_probability: {int(time.time())} | {tx_hash.hex()}")
    return tx_hash
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

#################### DATA

dataset = join(data_base_path, config['data']['dataset'])
df = pd.read_csv(dataset, nrows = num_rows_from_dataset)
X = df.drop(columns=['Activity', 'timestamp']).values
y = df['Activity'].replace(situ_int_mapping).values # replace string to int
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=0)
X_train = (StandardScaler().fit_transform(X_train)*100).astype(int)
X_test = (StandardScaler().fit_transform(X_test)*100).astype(int)

df_new = pd.DataFrame(X_train)
df_new['Activity'] = y_train

result_json['train_size'] = len(X_train)    #====
result_json['test_size'] = len(X_test)      #====

X_train_N = []
X_train_N_arr = []
for k, v in situ_int_mapping.items():
    X_train_N.append(df_new.loc[df_new['Activity']==v])
    X_train_N_arr.append(df_new.loc[df_new['Activity']==v].to_numpy())



train_tx_hashes = []
predict_tx_hashes = []
sensor_nonce = web3.eth.get_transaction_count(sensor_pbk, 'pending')
################# TRAINING
_means_N = []
_vars_N = []
# total_gas_used = 0
total_train_time = 0
for i in range(len(situ_int_mapping)):    # num classes
    _means_class = []
    _vars_class = []
    for j in range(X_train.shape[1]):   # num features
        print(f"Training class:{i}, feature:{j}")
        print(sensor_nonce)
        start_time = time.time()
        # gas_per_row = 0

                                # each feature col       # num rows for this target
        # tx_receipt = cal_mean(X_train_N_arr[i][:,j].tolist(), X_train_N[i].shape[0], sensor_pbk, sensor_pvk, train_contract)
        tx_hash = cal_mean(sensor_nonce, X_train_N_arr[i][:,j].tolist(), X_train_N[i].shape[0], sensor_pbk, sensor_pvk, train_contract)
        train_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        sensor_nonce += 1
        # gas_per_row += tx_receipt.gasUsed
        _mean_feature = train_contract.functions.get_mean().call()  # get mean at current stage
        _means_class.append(_mean_feature)
        
        # print(X_train_N_arr[i][:,j].tolist(), int(_means_class[j]), X_train_N[i].shape[0])
        # tx_receipt = cal_variance(X_train_N_arr[i][:,j].tolist(),int(_means_class[j]), X_train_N[i].shape[0], sensor_pbk, sensor_pvk, train_contract)
        tx_receipt = cal_variance(sensor_nonce, X_train_N_arr[i][:,j].tolist(),int(_means_class[j]), X_train_N[i].shape[0], sensor_pbk, sensor_pvk, train_contract)
        train_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        sensor_nonce += 1
        # gas_per_row += tx_receipt.gasUsed
        _var_feature = train_contract.functions.get_variance().call()   # get var at current stage
        _vars_class.append(_var_feature)
        
        # print(f"{i}{j} | mean: {int(_means_class[j])} | var: {var_}")
        # print(f"{i}{j} | actual mean: {np.mean(X_train_N_arr[i][:,j].tolist())} | actual var: {np.var(X_train_N_arr[i][:,j].tolist())}")
        # print("---")
        
        end_time = time.time()
        
        row_training_time = end_time - start_time
        training_times.append(row_training_time)
        total_train_time += row_training_time
    
        # training_gases.append(gas_per_row)
        # total_gas_used += gas_per_row
    
    _means_N.append(_means_class)
    _vars_N.append(_vars_class)

# print("TRAIN >>>>", total_gas_used)

result_json['train_time'] = total_train_time       #====
# result_json['train_gas'] = total_gas_used           #====

time.sleep(10)
################# TESTING
priorN = []
for i in range(len(situ_int_mapping)):    # num classes
    priorN.append( int(len(X_train_N_arr[i])/len(df_new)*100) )

# total_gas_used = 0
total_inference_time = 0
pred = np.zeros(X_test.shape[0], dtype=int)
for i in range(X_test.shape[0]):    # each row
    print(f"predicting row {i}")
    start_time = time.time()
    # gas_per_row = 0
    
    f = X_test[i,:].tolist()
    for j in range(len(situ_int_mapping)):    # each class
        # tx_receipt = cal_class(f, _vars_N[j], _means_N[j], sensor_pbk, sensor_pvk, predict_contract)
        # gas_per_row += tx_receipt.gasUsed
        tx_hash = cal_class(sensor_nonce, f, _vars_N[j], _means_N[j], sensor_pbk, sensor_pvk, predict_contract)
        predict_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        sensor_nonce += 1
        time.sleep(2)
    
    try:
    # tx_receipt = cal_probability(priorN[0],priorN[1],priorN[2],priorN[3],priorN[4],priorN[5], sensor_pbk, sensor_pvk, predict_contract)
    # gas_per_row += tx_receipt.gasUsed
        tx_hash = cal_probability(sensor_nonce, priorN[0],priorN[1],priorN[2],priorN[3],priorN[4],priorN[5], sensor_pbk, sensor_pvk, predict_contract)
        predict_tx_hashes.append( (int(time.time()), tx_hash.hex()) )
        sensor_nonce += 1
        time.sleep(2)
    except Exception as e:
        print(e)
        time.sleep(2)
        sensor_nonce += 1
    
    pred[i] = int( predict_contract.functions.predict().call() ) 
    end_time = time.time()
    
    inference_time_per_row = end_time - start_time
    inference_times.append(inference_time_per_row)
    total_inference_time += inference_time_per_row
    
    # inference_gases.append(gas_per_row)
    # total_gas_used += gas_per_row
    
    # print(f" pred: {pred[i]} | actual: {y_test[i]}")

# print("INFERENCE >>>>", total_gas_used)
# print("accuracy: ", accuracy_score(pred,y_test))

# result_json['avg_inference_gas'] = total_gas_used / len(X_test)         #====
result_json['avg_inference_time'] = total_inference_time / len(X_test)  #====
result_json['test_accuracy'] = accuracy_score(pred,y_test)              #====
    
print(json.dumps(result_json, indent=4))
with open(join(curr_dir, config['result']['bayes_result_sumary']), 'w') as file:
    json.dump(result_json, file, indent=4)

training_times = np.array(training_times)
training_gases = np.array(training_gases)
inference_times = np.array(inference_times)
inference_gases = np.array(inference_gases)

# Creating a dictionary with arrays
training_data = {
    'training_times': training_times,
    # 'training_gases': training_gases,
}
inference_data = {
    'inference_times': inference_times,
    # 'inference_gases': inference_gases
}

df_result = pd.DataFrame(training_data)
df_result.to_csv(join(curr_dir, config['result']['bayes_training_result_detail']), index=False)

df_result = pd.DataFrame(inference_data)
df_result.to_csv(join(curr_dir, config['result']['bayes_inference_result_detail']), index=False)


df_train = extract_receipt(train_tx_hashes)
df_predict = extract_receipt(predict_tx_hashes)

df_train.to_csv(join(curr_dir, df_train_f), index=False)
df_predict.to_csv(join(curr_dir, df_predict_f), index=False)

print(df_train)
print("---")
print(df_predict)