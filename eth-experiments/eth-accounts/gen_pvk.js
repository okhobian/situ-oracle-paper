var keythereum = require("keythereum");
var datadir = process.env.ETH_ACCOUNT_BASE_DIR;
var address= "0x76a3Ae1fD1eceA497B106e7Dda965a0245e29e31";
const password = "123";

var keyObject = keythereum.importFromFile(address, datadir);
var privateKey = keythereum.recover(password, keyObject);
console.log(privateKey.toString('hex'));