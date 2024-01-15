import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import json
from web3 import Web3

from os.path import dirname, join, abspath
import os



# Get the current directory where the script is located
current_directory = dirname(abspath(__file__))
train_build = join(current_directory, '../truffle-paper-example/build/contracts/TrainExample.json')
predict_build = join(current_directory, '../truffle-paper-example/build/contracts/PredictExample.json')


ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
print(web3.is_connected())

abi_train = json.load(open(train_build))
cont_t = "0x785aF53B59a376878057295a00fb0478F7d4acE3"
cont_train = web3.eth.contract(address=cont_t,abi=abi_train['abi'])

abi_pred = json.load(open(predict_build))
cont_p = "0xA5d33450A14EDf7FCdd5a4e45035e6909FCD9eb2"
cont_pred = web3.eth.contract(address=cont_p,abi=abi_pred['abi'])

#grit
account1 = "0xcB4E892748fdE791A0efE9C7a40E3510c598e0fd"
private_key1="45acbaf9e5fc998a38672a5f720a73839f47ddaec96835c11a182e053b3ed96e"


def cal_mean(feature, n, account, private_key, contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_train.functions.cal_mean(
        feature,n
    ).build_transaction({
        'gas': 3000000,
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
        'gas': 6000000,
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
        'gas': 6000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt


def cal_probability(A0,A1,B0,B1,C0,C1,D0,D1,Prior0,Prior1,account, private_key,contract):
    nonce = web3.eth.get_transaction_count(account)
    tx = cont_pred.functions.cal_exponent (
        A0,A1,B0,B1,C0,C1,D0,D1,Prior0,Prior1
    ).build_transaction({
        'gas': 6000000,
        'gasPrice': web3.to_wei('10','gwei'),
        'from': account,
        'nonce': nonce
    })
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

'''
def compare(A,B):
    if(A>B): return 0
    else: return 1
    

def mux1(v,k):
    x = 1
#     print(type(v))
    for i in range(len(v)):
        if(i==k): continue
        else:  x = x* v[i]
    return x

def compute_param(x,m,v):
    B0=1; C0 =1.0;  A0=1; e = 2.718
    c = np.zeros(len(v));
    for i in range(len(v)):
        B0 = B0* int(v[i])
        c[i] = (x[i]-m[i])*(x[i]-m[i])
    D0 = B0
    for i in range(len(c)):
        C0 = C0+c[i]*mux1(v,i) 
    ex = float(C0/D0)
    return (A0,B0,C0,D0)

def exponent(x, m1,m2, v1,v2, Prior0,Prior1):
        A0,B0,C0,D0 = compute_param(x,m1,v1)
        print(A0,B0,C0,D0)
        A1,B1,C1,D1 = compute_param(x,m2,v2)
        print(A1,B1,C1,D1)
        A = A0*B1
        B = B0*A1   
        C = (C0*D1)-(C1*D0)
        D = D0*D1

        Q = int(C/D)
        R = C%D;
       
        Enum=19
        Eden=7
        
        edenR = 24*(D*D*D*D)
        enumR = 24*D*D*D*D+ R*24*D*D*D+R*R*12*D*D+R*R*R*4*D+R*R*R*R
        
        if(Q>=0):
            P0 = A*1*edenR*Prior0*Prior0
            P1 = B*1000*enumR*Prior1*Prior1
        else:
            P0 = A*1000*edenR*Prior0*Prior0
            P1 = B*1*enumR*Prior1*Prior1
        if(P0>P1): 
            return 0
        else : 
            return 1

'''
# print(cont_train.functions.set_mean_var().call())


'''
# df = pd.read_csv("heart.csv")
# target = np.array(df['target'].values)
# features = np.array(df.drop(columns = ['target']))

# x_train,x_test, y_train, y_test = train_test_split(features,target,test_size=0.2, random_state=0)
# x_train = (StandardScaler().fit_transform(x_train)*100).astype(int)
# x_test = (StandardScaler().fit_transform(x_test)*100).astype(int)
# # # x_test

# # x_train.shape

# gnb = GaussianNB()
# y_pred = gnb.fit(x_train, y_train).predict(x_test)
# print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
# print(accuracy_score(y_test, y_pred))
'''


df = pd.read_csv("heart.csv")
target = np.array(df['target'].values)
features = np.array(df.drop(columns = ['target']))
dfp = df.groupby('target').size()   # how many times each unique value appears in the array
Prior0 = int(dfp[0]/len(df)*100)
Prior1 = int(dfp[1]/len(df)*100)

print("PRIORS", Prior0, Prior1)

# print(df.head())
# print(target)
# print(target.shape)
# print(features)
# print(features.shape)
# print(dfp)
# print(Prior0, Prior1)

x_train,x_test, y_train, y_test = train_test_split(features,target,test_size=0.2, random_state=0)
x_train = (StandardScaler().fit_transform(x_train)*100).astype(int)
x_test = (StandardScaler().fit_transform(x_test)*100).astype(int)

# print(x_train)
# print(x_test)

df_new = pd.DataFrame(x_train,columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13'])
df_new['target'] = y_train
# df_mean = df_new.groupby(['target']).mean().astype(int)
# df_var = df_new.groupby(['target']).var().astype(int)
# mean = df_mean.values
# var = df_var.values

# print(mean)
# print(var)

x_train_0 = df_new.loc[df_new['target']==0]
x_train_1 = df_new.loc[df_new['target']==1]
x_train_0_ar = x_train_0.to_numpy()
x_train_1_ar = x_train_1.to_numpy()

# print(x_train_0_ar)
# print(x_train_1_ar)

# print(x_train.shape[1])
# print(x_train_0_ar[:,13].tolist())


_means0=[]
_means1=[]
total_gas_used = 0
                # 13 features
for i in range(x_train.shape[1]):
                            # each feature col       # num rows for this target
    tx_receipt = cal_mean(x_train_0_ar[:,i].tolist(), x_train_0.shape[0], account1, private_key1,cont_train)
    total_gas_used += tx_receipt.gasUsed
    _means0.append(cont_train.functions.get_mean().call())
    
    tx_receipt = cal_mean(x_train_1_ar[:,i].tolist(), x_train_1.shape[0], account1, private_key1,cont_train)
    total_gas_used += tx_receipt.gasUsed
    _means1.append(cont_train.functions.get_mean().call())

print(_means0)
print("===")
print(_means1)

_var0=[]
_var1=[]
for i in range(x_train.shape[1]):        
    tx_receipt = cal_variance(x_train_0_ar[:,i].tolist(),int(_means0[i]), x_train_0.shape[0], account1, private_key1,cont_train)
    total_gas_used += tx_receipt.gasUsed
    _var0.append(cont_train.functions.get_variance().call())
    
    tx_receipt = cal_variance(x_train_1_ar[:,i].tolist(),int(_means1[i]), x_train_1.shape[0], account1, private_key1,cont_train)
    total_gas_used += tx_receipt.gasUsed
    _var1.append(cont_train.functions.get_variance().call())
    
print("~~~")
print(_var0)
print("===")
print(_var1)
print("TRAIN >>>>", total_gas_used)

# print(x_test[i,:].tolist())
# print(_means0)
# print(_var0)

Prior0 = int(len(x_train_0)/len(df_new)*100)
Prior1 = int(len(x_train_1)/len(df_new)*100)
print("PRIORS", Prior0, Prior1)

# var0_less = np.sqrt(np.array(_var0)).astype(int)
# var1_less = np.sqrt(np.array(_var1)).astype(int)

# print(var0_less.tolist())
# print(var1_less.tolist())
# type(var0_less)

pred = np.zeros(x_test.shape[0])
# e =2.718
for i in range(x_test.shape[0]):
    f = x_test[i,:].tolist()
    print(f)
    tx_receipt = cal_class(f,_var0,_means0, account1, private_key1,cont_pred)
    A0,B0,C0,D0 = cont_pred.functions.get_parameters().call()
    B0,C0,D0 = int(B0/100000000000000000000000000000000000000000000),int(C0/100000000000000000000000000000000000000000000),int(D0/100000000000000000000000000000000000000000000)
    # print("A0,B0,C0,D0", A0,B0,C0,D0)
#     print(A0/B0*pow(e, -(C0/D0))*pow(Prior0,2))

    tx_receipt = cal_class(f,_var1, _means1, account1, private_key1,cont_pred)
    A1,B1,C1,D1 = cont_pred.functions.get_parameters().call()
    B1,C1,D1 = int(B1/100000000000000000000000000000000000000000000),int(C1/100000000000000000000000000000000000000000000),int(D1/100000000000000000000000000000000000000000000)
    
    cal_probability(A0,A1,B0,B1,C0,C1,D0,D1,Prior0,Prior1, account1,private_key1,cont_pred)
    # print("A1,B1,C1,D1", A1,B1,C1,D1)
#     print(A1/B1*pow(e, -(C1/D1))*pow(Prior1,2))

    Y0, Y1 = cont_pred.functions.get_exponent().call()
    print(Y0,Y1)
    pred[i] = cont_pred.functions.compare().call() #compare(Y0,Y1) 
  
  
print("pred: \n", pred)

print("accuracy: ", accuracy_score(pred,y_test))