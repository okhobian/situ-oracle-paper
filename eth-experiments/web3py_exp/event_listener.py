from web3 import Web3
from web3.auto import w3
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import json
import time
import yaml
import pandas as pd
import asyncio
from os.path import dirname, join, abspath
from os import environ

actuation_times = []
event_counter = 0

#################### Callback functions
def handle_event(event):
    global event_counter 
    event_counter = event_counter + 1
    tx_receipt = web3.eth.wait_for_transaction_receipt(event['transactionHash'])
    result = Acutation_Event.process_receipt(tx_receipt)
    time_actuated = int(time.time())
    actuation_times.append(time_actuated)
    print( f"event {event_counter} | time_actuated:{time_actuated} | value:{result[0]['args']['values']}" )
    print("===")
    
async def log_loop(event_filter, poll_interval):
    while True:
        # for event in event_filter.get_new_entries():    # event_filter.get_all_entries
        for event in event_filter.get_all_entries():
            handle_event(event)
        await asyncio.sleep(poll_interval)

#################### Get the current directory & Load config file
truffle_base_dir = environ.get("TRUFFLE_BASE_DIR")
curr_dir = dirname(abspath(__file__))
with open(join(curr_dir, join(curr_dir, 'config.yml')), 'r') as file:
    config = yaml.safe_load(file)

#################### Init Chain
network_url = config['chain']['network_url']
chain_id =  config['chain']['chain_id']
web3 = Web3(HTTPProvider(network_url))
if config['chain']['is_poa']: w3.middleware_onion.inject(geth_poa_middleware, layer=0)    # specific for PoA
print("Network Connected: ", web3.is_connected())

#################### Init Situ contract
situ_contract_build = join(truffle_base_dir, config['chain']['situ_build'])
situ_contract_abi = json.load(open(situ_contract_build))['abi']
situ_contract_addr = json.load(open(situ_contract_build))['networks'][chain_id]['address']
situ_contract = web3.eth.contract(address=situ_contract_addr, abi=situ_contract_abi)
Acutation_Event = situ_contract.events.Actuation()

#################### MAIN
block_filter = web3.eth.filter({'fromBlock':'latest', 'address':situ_contract_addr})
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete( asyncio.gather( log_loop(block_filter, 1) ))
    
finally:
    loop.close()
    
    # Save context_update_times to file
    df = pd.DataFrame(actuation_times, columns=['actuation_times'])
    df.to_csv(join(curr_dir, config['result']['actuation_times']), index=False)