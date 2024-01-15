// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

contract Situ {
    string public current_situ;
    address public oracle_addr;
    mapping(string => string) public situ_based_actuations;   // e.g., watchtv => 01101001
    // event Actuation(uint256[] values);
    event Actuation(string values);

    constructor(address init_oracle_addr, string[] memory situ_list, string[] memory actuator_values) {
        require(situ_list.length == actuator_values.length);
        for (uint256 i = 0; i < situ_list.length; i++) {
            situ_based_actuations[situ_list[i]] = actuator_values[i];
        }
        current_situ = "other";
        oracle_addr = init_oracle_addr;
    }

    function update_situ(string memory new_situ) public {
        require(msg.sender == oracle_addr, "unidentified oracle address");
        current_situ = new_situ;
        string memory actuationValues = situ_based_actuations[new_situ];
        emit Actuation(actuationValues);
    }

    function get_situ() public view returns (string memory) {
        return current_situ;
    }

    // Function to transfer oracle node address
    function switch_oracle(address new_oracle_addr) public {
        require(msg.sender == oracle_addr, "unidentified oracle address");
        oracle_addr = new_oracle_addr;
    }
}