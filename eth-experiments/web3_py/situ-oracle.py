import os
import binascii
import json
import web3
from web3.auto import w3
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware

# connect to JSON RPC node
w3 = Web3(HTTPProvider('http://192.168.66.128:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)    # specific for POA

# get contract ABI and Addr (compiled via Truffle)
f = open('./truffle/build/contracts/Situ.json')
contract_json = json.load(f)
contract_addr = contract_json['networks']['61857']['address']
contract_abi = contract_json['abi']
situContract = w3.eth.contract(address=contract_addr, abi=contract_abi)

# compose transaction
nonce = w3.eth.get_transaction_count('0xE49c484dF14208A6E3413A991a79013fae145A26')
transaction = situContract.functions.updateSitu("sleep").buildTransaction(
    {
        # 'maxFeePerGas': 0, 
        # 'maxPriorityFeePerGas': 0,
        # 'type': 0x0,
        # 'gas': 70000,
        'nonce': nonce,
        'chainId': 61857,
        'gasPrice': w3.toWei('1', 'gwei'),
        'from': "0xE49c484dF14208A6E3413A991a79013fae145A26",
    }
)

# get private key to the account, not safe way, only for demostration
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, "keystore.txt")) as keyfile:
    encrypted_key = keyfile.read()
    private_key = w3.eth.account.decrypt(encrypted_key, '12345')
    private_key = binascii.b2a_hex(private_key)

print('nonce: ', w3.eth.get_transaction_count('0xE49c484dF14208A6E3413A991a79013fae145A26'))
print('public key: ', "0xE49c484dF14208A6E3413A991a79013fae145A26")
print('private key: ', private_key.decode())
print(transaction)

# sign and send transaction
signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key.decode())
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f'Tx successful with hash: { tx_receipt.transactionHash.hex() }')