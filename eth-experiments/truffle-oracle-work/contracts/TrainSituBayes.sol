// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

/** 
    Example code taken from BCCA'22 paper "Making Smart Contracts Predict and Scale"
    https://github.com/syber2020/Smart-Contract-To-Predict
    -- modified to take 29 sensory features, and classify 6 situation classes

    [1] S. Badruddoja, R. Dantu, Y. He, M. Thompson, A. Salau, and K. Upadhyay, 
    “Making Smart Contracts Predict and Scale,” 
    in 2022 Fourth International Conference on Blockchain Computing and Applications (BCCA), 
    San Antonio, TX, USA: IEEE, Sep. 2022, pp. 127–134. doi: 10.1109/BCCA55292.2022.9922480.

    [2] S. Badruddoja, R. Dantu, Y. He, K. Upadhayay, and M. Thompson, 
    “Making Smart Contracts Smarter,” 
    in 2021 IEEE International Conference on Blockchain and Cryptocurrency (ICBC), 
    Sydney, Australia: IEEE, May 2021, pp. 1–3. doi: 10.1109/ICBC51069.2021.9461148.
 */

contract TrainSituBayes {

    int mean; int variance;

    function cal_mean(int[] memory features, int n) public{
        int sum=0; mean=0; 

        if (n != 0){
            for(uint i=0; i< features.length; i++){
                sum = sum + features[i];
            }
            mean = sum / n;
        }
        
    }

    function cal_variance(int[] memory features, int m, int n) public{
        int sum=0; variance=0;

        if (n != 0){
            for(uint i=0; i<features.length; i++){
                sum = sum + (features[i]-m)*(features[i]-m);
            }
            variance = sum / n;
        }
    }

    function get_mean() public view returns(int){
        return mean;
    }

    function get_variance() public view returns(int){
        return variance;
    }
}