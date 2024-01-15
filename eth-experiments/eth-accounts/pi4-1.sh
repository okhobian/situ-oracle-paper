# pi4-1
nohup geth 	--nousb \
            --datadir=$PWD \
            --syncmode 'full' \
            --port 30305 \
            --nat extip:192.168.0.251 \
            --networkid 4935 \
            --miner.gasprice 0 \
            --miner.gastarget 470000000000 \
            --http \
            --http.addr 0.0.0.0 \
            --http.corsdomain '*' \
            --http.port 8545 \
            --http.vhosts '*' \
            --http.api admin,eth,miner,net,txpool,personal,web3 \
            --miner.etherbase="0x45936e885c5f147363c05737Bd4cF8C70324d1E9" \
            --mine \
            --allow-insecure-unlock \
            --unlock "0x45936e885c5f147363c05737Bd4cF8C70324d1E9,0x3CF03eF46d2b18e556e4d8E61644307576241b29,0xcC4DEeeD9b83E02E213EbC12d528e0fb3fEe9448,0x9b82F280DA04b39F7C252F1507E2Bb28cF205433,0x5401b84584237A2783d10252299ac269030cb742,0xEC831d1D15840F33d5966c222C5fB730EB10b2F4" \
            --password password.txt &
        
echo "Node pi4-1 Start"