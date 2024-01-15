// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

contract Context {
    mapping(string => uint256) public sensor_val;

    constructor(string[] memory sensor_list) 
    {
        // Iterate through the list of strings and set initial values to 0 for each key (sensor)
        for (uint256 i = 0; i < sensor_list.length; i++) {
            string memory key = sensor_list[i];
            sensor_val[key] = 0;
        }
    }

    function update_context_value(string memory _sensor, uint256 _newValue) public {
        sensor_val[_sensor] = _newValue;
    }

    // Function to retrieve the FixedArray for a specific address
    function get_context_value(string memory _sensor) public view returns (uint256) {
        return sensor_val[_sensor];
    }
}