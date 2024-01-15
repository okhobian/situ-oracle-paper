from web3 import Web3
# from web3.auto import w3
import json
import argparse
from threading import Thread
import time

# Replace with your Ethereum node URL (or use Infura)
ethereum_node_url = "http://127.0.0.1:7545"

# Create a Web3.py instance
w3 = Web3(Web3.HTTPProvider(ethereum_node_url))

Hello     ="0x74143BEcEAb9f85004073e51C73954B0e8dA55c8"
Context   ="0xcda5888058eDdFFCf8768911e7e7e9410A90219e"
Situ      ="0x5D9155A9c88e832440faC3EBEF330aFe6EBCBA8f"

def test_context():
    
    # get contract ABI and Addr (compiled via Truffle)
    f = open('../truffle/build/contracts/Context.json')
    contract_json = json.load(f)
    contract_addr = Context
    contract_abi = contract_json['abi']
    contextContract = w3.eth.contract(address=contract_addr, abi=contract_abi)

    # Replace with the actual function name and arguments
    tx_hash = contextContract.functions.updateContextValues("bathroomLight", 0).transact({'from': "0xc3b4C0958E3c753953BD9C493cC354d845e8c1c9"})

    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # You can also read data from the contract
    value = contextContract.functions.getValue("bathroomLight").call()
    print(value)
    

def test_situ():
    
    # get contract ABI and Addr (compiled via Truffle)
    f = open('../truffle/build/contracts/Situ.json')
    contract_json = json.load(f)
    contract_addr = Situ
    contract_abi = contract_json['abi']
    situContract = w3.eth.contract(address=contract_addr, abi=contract_abi)        

    cur_situ = situContract.functions.get_situ().call()
    print(cur_situ)

    # must from oracle address specified in 4_deploy_situ.js
    tx_hash = situContract.functions.update_situ("rest").transact({'from': "0xc3b4C0958E3c753953BD9C493cC354d845e8c1c9"})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    new_situ = situContract.functions.get_situ().call()
    print(new_situ)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run some functions')

    # Add a command
    parser.add_argument('--func')

    # Get our arguments from the user
    args = parser.parse_args()

    if args.func == 'context':
        test_context()

    if args.func == 'situ':
        test_situ()