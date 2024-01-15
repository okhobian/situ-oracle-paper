// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

contract TrainExample {

    int mean; int variance;

    function cal_mean(int[] memory features, int n) public{
        int sum=0; mean=0; 
        for(uint i=0; i< features.length; i++){
            sum = sum + features[i];
        }
        mean = sum / n;
    }

    function cal_variance(int[] memory features, int m, int n) public{
        int sum=0;
        for(uint i=0; i<features.length; i++){
            sum = sum + (features[i]-m)*(features[i]-m);
        }
        variance = sum / n;
    }

    function get_mean() public view returns(int){
        return mean;
    }

    function get_variance() public view returns(int){
        return variance;
    }
}