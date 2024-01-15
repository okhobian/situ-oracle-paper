import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import json
import time
from web3 import Web3

from os.path import dirname, join, abspath
import os

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
ACTIVITIES = {'sleep':0, 'eat':1, 'other':2, 'work':3, 'leisure':4, 'personal':5}

TRAIN_CONTRACT = "TrainSituBayes"
PREDICT_CONTRACT = "PredictSituBayes"

# Get the current directory where the script is located
data_base_path = os.environ.get("OPENSHS_DATA_PATH")
current_directory = dirname(abspath(__file__))
train_build = join(current_directory, f"../truffle-paper-example/build/contracts/{TRAIN_CONTRACT}.json")
predict_build = join(current_directory, f"../truffle-paper-example/build/contracts/{PREDICT_CONTRACT}.json")

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
print(web3.is_connected())

abi_train = json.load(open(train_build))
cont_t = "0x9d227C5B044dD3b44d55e8D7435E5dbed33D9400"
cont_train = web3.eth.contract(address=cont_t,abi=abi_train['abi'])

abi_pred = json.load(open(predict_build))
cont_p = "0xaD1a3D262fd87bF7A6B4A402D7cA4f95f36a0eB3"
cont_pred = web3.eth.contract(address=cont_p,abi=abi_pred['abi'])

account1 = "0x0A27605fC3b321C4D4F1f6DecDE170105eEc61E8"
private_key1="75232a0277cee80bd1fee130f49b7d5ef3719ce7214b3d4529e0d0e0e10c7b37"


def cal_mean(feature, n, account, private_key, contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_train.functions.cal_mean(
        feature,n
    ).build_transaction({
        'gas': 30000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def cal_variance(feature, mean, n, account, private_key,contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_train.functions.cal_variance(
        feature,mean,n
    ).build_transaction({
        'gas': 60000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt
    
def cal_class(features, variances, means, account, private_key, contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_pred.functions.cal_parameters(
        features ,variances,means 
    ).build_transaction({
        'gas': 60000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def cal_probability(Prior0, Prior1, Prior2, Prior3, Prior4, Prior5,account, private_key,contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_pred.functions.cal_probability (
        Prior0, Prior1, Prior2, Prior3, Prior4, Prior5
    ).build_transaction({
        'gas': 60000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt


datasets = []
for x in range(2,3):
# for x in [1,5,2,7,6]:
    datasets.append(data_base_path + f'd{x}_1m_0tm.csv')

for dataset in datasets:
    dataset_name = dataset.split('/')[-1]
    result_json['dataset'] = dataset_name   #====
    print(f"Training {dataset_name}")
    
    ## sub-dataset for contract vaildation
    # columns_to_load = ['wardrobe','tv','oven','Activity', 'timestamp']
    # df = pd.read_csv(dataset, usecols=columns_to_load, nrows=3000)
    df = pd.read_csv(dataset, nrows=5)
    print(df)
    # df = pd.read_csv(dataset)
    df['Activity'] = df['Activity'].replace(ACTIVITIES) # replace string to int

    # each row at once
    X = df.drop(columns=['Activity', 'wardrobe', 'timestamp']).values
    y = df['Activity'].values
    dfp = df.groupby('Activity')
    dfps = dfp.size()   # how many times each unique value appears in the array
        
    # Splitting the dataset using TimeSeriesSplit
    # tscv = TimeSeriesSplit(n_splits=2)
    # train_index, test_index = list(tscv.split(X))[0]
    # X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    # y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0)
    X_train = (StandardScaler().fit_transform(X_train)*100).astype(int)
    X_test = (StandardScaler().fit_transform(X_test)*100).astype(int)
    
    df_new = pd.DataFrame(X_train)
    df_new['Activity'] = y_train
    
    result_json['train_size'] = len(X_train)    #====
    result_json['test_size'] = len(X_test)      #====
    
    X_train_N = []
    X_train_N_arr = []
    for k, v in ACTIVITIES.items():
        X_train_N.append(df_new.loc[df_new['Activity']==v])
        X_train_N_arr.append(df_new.loc[df_new['Activity']==v].to_numpy())
        
    # print(X_train_N_arr[1])
    # # print(X_train.shape[1])
    # print(X_train_N_arr[1].shape[0])
    # print(X_train_N_arr[1][:,2].tolist())
    
    ################# TRAINING
    _means_N = []
    _vars_N = []
    total_gas_used = 0
    total_train_time = 0
    for i in range(len(ACTIVITIES)):    # num classes
        _means_class = []
        _vars_class = []
        for j in range(X_train.shape[1]):   # num features
            print(f"Training class:{i}, feature:{j}")
            start_time = time.time()
            gas_per_row = 0

                                    # each feature col       # num rows for this target
            tx_receipt = cal_mean(X_train_N_arr[i][:,j].tolist(), X_train_N[i].shape[0], account1, private_key1,cont_train)
            gas_per_row += tx_receipt.gasUsed
            _mean_feature = cont_train.functions.get_mean().call()  # get mean at current stage
            _means_class.append(_mean_feature)
            
            
            # print(X_train_N_arr[i][:,j].tolist(), int(_means_class[j]), X_train_N[i].shape[0])
            tx_receipt = cal_variance(X_train_N_arr[i][:,j].tolist(),int(_means_class[j]), X_train_N[i].shape[0], account1, private_key1,cont_train)
            gas_per_row += tx_receipt.gasUsed
            _var_feature = cont_train.functions.get_variance().call()   # get var at current stage
            _vars_class.append(_var_feature)
            
            # print(f"{i}{j} | mean: {int(_means_class[j])} | var: {var_}")
            # print(f"{i}{j} | actual mean: {np.mean(X_train_N_arr[i][:,j].tolist())} | actual var: {np.var(X_train_N_arr[i][:,j].tolist())}")
            # print("---")
            
            end_time = time.time()
            
            ### test accuracy at current stage
            
            
        
            row_training_time = end_time - start_time
            training_times.append(row_training_time)
            total_train_time += row_training_time
        
            training_gases.append(gas_per_row)
            total_gas_used += gas_per_row
        
        _means_N.append(_means_class)
        _vars_N.append(_vars_class)

    # print(_means_N)
    # print(_vars_N)
    print("TRAIN >>>>", total_gas_used)

    result_json['train_time'] = total_train_time       #====
    result_json['train_gas'] = total_gas_used           #====
    

    ################# TESTING
    priorN = []
    for i in range(len(ACTIVITIES)):    # num classes
        priorN.append( int(len(X_train_N_arr[i])/len(df_new)*100) )
    
    print("priorN: ", priorN)
    
    total_gas_used = 0
    total_inference_time = 0
    pred = np.zeros(X_test.shape[0], dtype=int)
    for i in range(X_test.shape[0]):    # each row
        print(f"predicting row {i} >> ", end='')
        start_time = time.time()
        gas_per_row = 0
        
        f = X_test[i,:].tolist()
        for j in range(len(ACTIVITIES)):    # each class
            tx_receipt = cal_class(f, _vars_N[j], _means_N[j], account1, private_key1, cont_pred)
            gas_per_row += tx_receipt.gasUsed
        
        tx_receipt = cal_probability(priorN[0],priorN[1],priorN[2],priorN[3],priorN[4],priorN[5], account1,private_key1,cont_pred)
        gas_per_row += tx_receipt.gasUsed
        
        pred[i] = int( cont_pred.functions.predict().call()  ) 
        end_time = time.time()
        
        inference_time_per_row = end_time - start_time
        inference_times.append(inference_time_per_row)
        total_inference_time += inference_time_per_row
        
        inference_gases.append(gas_per_row)
        total_gas_used += gas_per_row
        
        # print(f"{i} >> pred: {pred[i]} | actual: {y_test[i]}")
        print(f" pred: {pred[i]} | actual: {y_test[i]}")

    print("accuracy: ", accuracy_score(pred,y_test))
    
    result_json['avg_inference_gas'] = total_gas_used / len(X_test)         #====
    result_json['avg_inference_time'] = total_inference_time / len(X_test)  #====
    result_json['test_accuracy'] = accuracy_score(pred,y_test)              #====
    
print(json.dumps(result_json, indent=4))
# df_result = pd.DataFrame(result_json)
# df_result.tocsv('result_sumary.csv', index=False)

with open('result_sumary.json', 'w') as file:
    json.dump(result_json, file, indent=4)

training_times = np.array(training_times)
training_gases = np.array(training_gases)
inference_times = np.array(inference_times)
inference_gases = np.array(inference_gases)

# Creating a dictionary with arrays
training_data = {
    'training_times': training_times,
    'training_gases': training_gases,
}
inference_data = {
    'inference_times': inference_times,
    'inference_gases': inference_gases
}

df_result = pd.DataFrame(training_data)
df_result.to_csv('training_result_detail.csv', index=False)

df_result = pd.DataFrame(inference_data)
df_result.to_csv('inference_result_detail.csv', index=False)

# f= [0, 0, 0, 0, 0, 0, 0, 23, -17, 0, 0, 0, 0, 0, 0, -166, 0, 0, 0, 0, 197, 0, 0, 0, -148, 0, 190, -152] 
# var= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6250, 30371, 112763, 0, 0, 0, 0] 
# mean= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -27, 0, -298, 208, -128, 192, -170, 0, -43, -191]
# tx_receipt = cal_class(f, var, mean, account1, private_key1, cont_pred)
# cal_probability(6,0,93,0,0,0, account1,private_key1,cont_pred)
