const Situ = artifacts.require("Situ");
const Context = artifacts.require("Context");
const SituDistributed = artifacts.require("SituDistributed");
const ContextDistributed = artifacts.require("ContextDistributed");
const TrainSituBayes = artifacts.require("TrainSituBayes");
const PredictSituBayes = artifacts.require("PredictSituBayes");

/** for situ-oracle */
var oracle_addr = "0xcC4DEeeD9b83E02E213EbC12d528e0fb3fEe9448";

/** for distributed authorities enmulation */
var init_model_hashes = [
  "334f4b3eb7f1b3715de7522369ee2d2b56651aaf31ed95f1cfb1676ac02c3f38", // gnb.joblib
  "892f0d1b58ef9c6ead436bc81bed857659ac219e3702cc0487c98667491ff4c4", // bnb.joblib
  "413d723a575182f2b5b3dc1ed94bbecc0480654e29949b6238df24c01cf83277", // mnb.joblib
  "3e60c6c81fc964b7eed4b3d92956ee4fc8d5ffc7dbd01caa6871d0d5e38b6c4d", // rnn.h5
  "c886d99b324f63be106082053a96598e31ad10e7e8d3eeba9741760fbf955c22", // lstm.h5
  "ed6d968a0d9095075723d54f3702a765d21119038d966048ed3a8d7c3640b0da", // gru.h5
]; 

var authority_addrs = [
  "0x9b82F280DA04b39F7C252F1507E2Bb28cF205433",
  "0x5401b84584237A2783d10252299ac269030cb742", 
  "0xEC831d1D15840F33d5966c222C5fB730EB10b2F4"
];

var situ_list = ["sleep", "eat", "work", "leisure", "personal", "other"]; 

var sensor_list = [ 
  "wardrobe", "tv", "oven", "officeLight", "officeDoorLock",	"officeDoor",	"officeCarp",	
  "office",	"mainDoorLock", "mainDoor",	"livingLight", "livingCarp", "kitchenLight",	
  "kitchenDoorLock", "kitchenDoor", "kitchenCarp", "hallwayLight", "fridge",	"couch", 
  "bedroomLight",	"bedroomDoorLock", "bedroomDoor", "bedroomCarp", "bedTableLamp", "bed", 
  "bathroomLight",	"bathroomDoorLock",	"bathroomDoor",	"bathroomCarp"
]; 

var actuator_values = [
  "01101001", // enmulate actuation for sleep
  "10010110", // enmulate actuation for eat
  "10111010", // enmulate actuation for work
  "01010101", // enmulate actuation for leisure
  "00101011", // enmulate actuation for personal
  "11000100"  // enmulate actuation for other
];

/** deploy contracts */
module.exports = function (deployer) {
  deployer.deploy(Situ, oracle_addr, situ_list, actuator_values);
  deployer.deploy(Context, sensor_list);
  deployer.deploy(SituDistributed, init_model_hashes, authority_addrs, situ_list, actuator_values);
  deployer.deploy(ContextDistributed, sensor_list);
  deployer.deploy(TrainSituBayes);
  deployer.deploy(PredictSituBayes);
};