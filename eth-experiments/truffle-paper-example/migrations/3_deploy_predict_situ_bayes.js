const Predict = artifacts.require("PredictSituBayes");

module.exports = function (deployer) {
  deployer.deploy(Predict);
};