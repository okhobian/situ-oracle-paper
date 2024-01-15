const Train = artifacts.require("TrainSituBayes");

module.exports = function (deployer) {
  deployer.deploy(Train);
};