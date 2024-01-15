const Predict = artifacts.require("PredictExample");

module.exports = function (deployer) {
  deployer.deploy(Predict);
};