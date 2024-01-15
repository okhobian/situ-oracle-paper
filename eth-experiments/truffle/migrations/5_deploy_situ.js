const Situ = artifacts.require("Situ");

var oracle_addr = "0x634c45E0926382307bBF316FB222f9bfbD1c251A";
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
  deployer.deploy(Situ, oracle_addr, situ_list, actuator_values);
};