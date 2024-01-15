// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

contract ContextDistributed {
    mapping(string => uint256[15]) private map;

    constructor(string[] memory sensor_list) 
    {
        // Iterate through the list of strings and set initial values to 0 for each key
        for (uint256 i = 0; i < sensor_list.length; i++) {
            string memory key = sensor_list[i];

            // Initialize a fixed-size integer array with zeros
            uint256[15] memory initialValue;
            for (uint256 j = 0; j < 15; j++) {
                initialValue[j] = 0;
            }

            map[key] = initialValue;
        }
    }

    function updateContextValues(string memory _sensor, uint256 _newValue) public {

        uint256[15] storage sensor_values = map[_sensor];
        
        // Shift the existing elements to the right
        for (uint256 i = sensor_values.length - 1; i > 0; i--) {
            sensor_values[i] = sensor_values[i - 1];
        }
        
        // Update the first element with the new value
        sensor_values[0] = _newValue;
    }

    // Function to retrieve the FixedArray for a specific address
    function getValue(string memory _sensor) public view returns (uint256[15] memory) {
        return map[_sensor];
    }

}