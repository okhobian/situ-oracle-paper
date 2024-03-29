// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;
// pragma solidity 0.5.10;

// for 13 feature and binary class
contract PredictExample{

    int prior1;int prior2;int prior3;int prior4; 
    int P0;int P1;
    int edenR;int enumR;
    int A0; int A1;
    int B0; int B1;
    int C0; int C1;
    int D0; int D1;
    int Q;int R;
    int a;int b;int d;
    int c1;int c2;int c3;int c4;
    int A;int B;int C;int D;
    int x;int y; 
    int Y0; int Y1;
    //int c;
    int mean;

    function mux(int[] memory v, uint k) public pure returns(int){
        int temp=1;
        for (uint i =0; i <v.length; i++){
            if(i==k){ continue;}
            else{temp = temp * v[i];}
        }
        return temp;
    }

    function cal_parameters(int[] memory features, int[] memory v, int[] memory m) public {
        A0=1; B0=1; D0=1; C0=1; 
        // int[] memory c;
        int[] memory c = new int[](features.length);
        for (uint i =0; i < features.length; i++ )
        {
            B0 = B0*v[i];
            c[i] = (features[i]-m[i])*(features[i]-m[i]);
        }
        for (uint i =0; i<c.length; i++){
            C0 = C0 + c[i]* mux(v, i); 
        }
        D0 = B0;
    }

    function get_parameters() public view returns(int,int,int,int){
        return(A0,B0,C0,D0);
    }

    // save all the gasussian probability computations for all classes and computer probabilty values
    function cal_exponent(int e0, int e1, int f0, int f1, int g0, int g1, int h0, int h1, int Prior0,int Prior1)public {
        A = e0*f1;
        B = f0*e1;
        C = (g0*h1)-(g1*h0);
        D = h0*h1;
        Q = int(C/D);
        R = C%D;
        edenR = 24*(D*D*D*D);
        enumR = 24*D*D*D*D+ R*24*D*D*D+R*R*12*D*D+R*R*R*4*D+R*R*R*R;
        if(Q>=0){
            Y0 = A*1*edenR*Prior0*Prior0;
            Y1 = B*1000*enumR*Prior1*Prior1;
        }else{
            Y0 = A*1000*edenR*Prior0*Prior0;
            Y1 = B*1*enumR*Prior1*Prior1;
        }
    }
    function get_exponent() public view returns (int, int){
        return (Y0,Y1);
    }

    function compare()public view returns(int){
        // int p0=0; int p1=1;
        if(Y0 > Y1)return(0);
        else return(1);
    }  
}