const SituDistributed = artifacts.require("SituDistributed");

var init_model_hashes = [
  "d0776360f4b3103c1859e170f42251319a588701691574e60d19cec1ab1d6180", // situ_gnb.joblib
  "3e60c6c81fc964b7eed4b3d92956ee4fc8d5ffc7dbd01caa6871d0d5e38b6c4d", // rnn.h5
  "c886d99b324f63be106082053a96598e31ad10e7e8d3eeba9741760fbf955c22", // lstm.h5
  "ed6d968a0d9095075723d54f3702a765d21119038d966048ed3a8d7c3640b0da", // gru.h5
]; 

var authority_addrs = [
  "0x634c45E0926382307bBF316FB222f9bfbD1c251A",
  "0x87cf816E0f32EC0B0Fa1cC514E034CEe1068F5B9", 
  "0x21971cB2C13621AB69f212884D6DCF358928b3f4"
];

var situ_list = ["sleep", "eat", "work", "leisure", "personal", "other"]; 

var actuator_values = [
  "01101001",
  "10010110",
  "10111010",
  "01010101",
  "00101011",
  "11000100"
];  // assume each row corresponding to each situation

module.exports = function (deployer) {
  deployer.deploy(SituDistributed, init_model_hash, authority_addrs, situ_list, actuator_values);
};