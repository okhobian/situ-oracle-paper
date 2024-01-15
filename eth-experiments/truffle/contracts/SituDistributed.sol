// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;

contract SituDistributed {
    string public current_situ;
    // string public model_hash;
    string [] public valid_model_hashes;
    address[] public authority_addrs;
    mapping(address => string) public authority_decisions;
    mapping(string => string) public situ_based_actuations;   // e.g., watchtv => 01101001
    event Actuation(string values);

    constructor(string[] memory init_model_hashes, address[] memory init_authority_addrs, string[] memory situ_list, string[] memory actuator_values) 
    {
        require(situ_list.length == actuator_values.length);

        // Init situ prediction model hash with keccak256
        // model_hash = init_model_hash;
        for (uint i = 0; i < init_model_hashes.length; i++) {
            valid_model_hashes.push(init_model_hashes[i]);
        }

        // Populate authority addresses
        for (uint256 i = 0; i < init_authority_addrs.length; i++) {
            authority_addrs.push(init_authority_addrs[i]);
        }

        // Init authority_decisions
        for (uint256 i = 0; i < init_authority_addrs.length; i++) {
            authority_decisions[authority_addrs[i]] = "none";
        }

        // Assign actuation values to each situation
        for (uint256 i = 0; i < situ_list.length; i++) {
            situ_based_actuations[situ_list[i]] = actuator_values[i];
        }

        // Random current situ to begin with
        current_situ = "other";
    }

    function update_situ(string memory new_situ, string memory local_model_hash) public 
    {
        require(isAuthority(msg.sender), "Sender is not an authority");
        // require(keccak256(bytes(local_model_hash)) == keccak256(bytes(model_hash)), "Suspicious local model");
        require(isValidModel(local_model_hash), "Suspicious local model");

        authority_decisions[msg.sender] = new_situ;
        if (allDecisionsMade()) 
        {
            // assume decisions are the same for now
            // which should since using the same model
            current_situ = new_situ;    
            string memory actuationValues = situ_based_actuations[new_situ];
            emit Actuation(actuationValues);
            clearDecisions();
        }
    }

    function get_situ() public view returns (string memory) {
        return current_situ;
    }


    // Function to check if an address belongs to an authority
    function isAuthority(address _address) internal view returns (bool) {
        for (uint i = 0; i < authority_addrs.length; i++) {
            if (authority_addrs[i] == _address) {
                return true;
            }
        }
        return false;
    }

    // Function to check if decisions are made by all authorities
    function allDecisionsMade() internal view returns (bool) {
        for (uint i = 0; i < authority_addrs.length; i++) {
            if (keccak256(bytes(authority_decisions[authority_addrs[i]])) == keccak256(bytes("none"))) {
                return false; // Found a decision that is "none"
            }
        }
        return true; // No decisions are "none"
    }

    // Function to check if a model hash is in valid
    function isValidModel(string memory _model_hash) internal view returns (bool) {
        for (uint i = 0; i < valid_model_hashes.length; i++) {
            if (keccak256(bytes(valid_model_hashes[i])) == keccak256(bytes(_model_hash))) {
                return true;
            }
        }
        return false;
    }

    // Function to clear decisions made by all authorities after update situ
    function clearDecisions() internal {
        for (uint i = 0; i < authority_addrs.length; i++) {
            authority_decisions[authority_addrs[i]] = "none";
        }
    }
}