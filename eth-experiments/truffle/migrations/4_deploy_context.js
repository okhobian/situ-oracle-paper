const Context = artifacts.require("Context");

var sensor_list = ["wardrobe", "tv", "oven", "officeLight", "officeDoorLock",	"officeDoor",	
  "officeCarp",	"office",	"mainDoorLock", "mainDoor",	"livingLight", "livingCarp",	
  "kitchenLight",	"kitchenDoorLock", "kitchenDoor", "kitchenCarp", "hallwayLight",	
  "fridge",	"couch", "bedroomLight",	"bedroomDoorLock", "bedroomDoor", "bedroomCarp",	
  "bedTableLamp",	"bed", "bathroomLight",	"bathroomDoorLock",	"bathroomDoor",	"bathroomCarp"
]; 

module.exports = function (deployer) {
  deployer.deploy(Context, sensor_list);
};