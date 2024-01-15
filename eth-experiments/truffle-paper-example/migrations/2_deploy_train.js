const Train = artifacts.require("TrainExample");

module.exports = function (deployer) {
  deployer.deploy(Train);
};