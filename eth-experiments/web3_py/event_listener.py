from web3 import Web3
# from web3.auto import w3
import json
import argparse
from threading import Thread
import time
import pandas as pd
import asyncio
from os.path import dirname, join, abspath

actuation_times = []

# Get the current directory where the script is located
current_directory = dirname(abspath(__file__))
dist_situ_build = join(current_directory, '../truffle/build/contracts/SituDistributed.json')
dist_context_build = join(current_directory, '../truffle/build/contracts/ContextDistributed.json')

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
print(web3.is_connected())

abi_situ = json.load(open(dist_situ_build))
cont_s = "0x0D9148b6E7Fb4B4ab4255846b67592573ee5f048"
cont_situ = web3.eth.contract(address=cont_s,abi=abi_situ['abi'])

abi_context = json.load(open(dist_context_build))
cont_c = "0xe61daB51f15A6B854AC071109BEfD2c4D6C5f7f5"
cont_context = web3.eth.contract(address=cont_c,abi=abi_context['abi'])

Acutation_Event = cont_situ.events.Actuation()


def handle_event(event):
    tx_receipt = web3.eth.wait_for_transaction_receipt(event['transactionHash'])
    result = Acutation_Event.process_receipt(tx_receipt)
    time_actuated = int(time.time())
    actuation_times.append(time_actuated)
    print(time_actuated, result[0]['args']['values'])
    
async def log_loop(event_filter, poll_interval):
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
            print("===")
        await asyncio.sleep(poll_interval)

def main():
    ethereum_node_url = "http://127.0.0.1:7545"
    w3 = Web3(Web3.HTTPProvider(ethereum_node_url))
    block_filter = w3.eth.filter({'fromBlock':'latest', 'address':cont_s})
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            asyncio.gather( log_loop(block_filter, 2) )
        )
    finally:
        loop.close()
        
         # Save context_update_times to file
        df = pd.DataFrame(actuation_times, columns=['actuation_times'])
        filename = 'actuation_times.csv'
        df.to_csv(filename, index=False)

if __name__ == '__main__':
    main()