// SPDX-License-Identifier: MIT
pragma solidity ^0.8.12;
// pragma solidity 0.5.10;

// for 29 feature and 6 classes
contract PredictSituBayes {

    uint constant num_situ = 6;

    // int prior1;int prior2;int prior3;int prior4; 
    int P0; int P1; int P2; int P3; int P4; int P5;
    
    int[num_situ] AX;
    int[num_situ] BX;
    int[num_situ] CX;
    int[num_situ] DX;
    // int A0; int A1; int A2; int A3; int A4; int A5;
    // int B0; int B1; int B2; int B3; int B4; int B5; 
    // int C0; int C1; int C2; int C3; int C4; int C5; 
    // int D0; int D1; int D2; int D3; int D4; int D5;
    
    int A; int B; int C;int D; int Q; int R; int edenR; int enumR; // exponent
    int Y0; int Y1;

    // int a; int b; int c; int d;
    // int c1;int c2;int c3;int c4;
    
    // int X1; int X2;int Y2;
    // int E; int F; int G; int H;
    // int X01; int X23; int X45;
    

    // int prior1;int prior2;int prior3;int prior4; 
    // int P0;int P1;
    // int edenR;int enumR;
    // int A0; int A1;
    // int B0; int B1;
    // int C0; int C1;
    // int D0; int D1;
    // int Q;int R;
    // int a;int b;int d;
    // int c1;int c2;int c3;int c4;
    // int A;int B;int C;int D;
    // int x;int y; 
    // int Y0; int Y1;
    // //int c;
    // int mean;

    function mux(int[] memory v, uint k) public pure returns(int)
    {
        int temp=1;
        for (uint i=0; i<v.length; i++)
        {
            if(i==k){ continue; }
            else{ temp = temp * v[i]; }
        }
        return temp;
    }

    function cal_parameters(int[] memory features, int[] memory v, int[] memory m) public 
    {
        for (uint i=0; i<num_situ; i++) 
        {
            AX[i] = 1;

            int[] memory c = new int[](features.length);
            for (uint j=0; j<features.length; j++)
            {
                BX[i] = 1; 
                DX[i] = 1;
                BX[i] = BX[i] * v[j];
                DX[i] = DX[i] * v[j];
                c[j] = (features[j]-m[j]) * (features[j]-m[j]);
            }
            for (uint j=0; j<features.length; j++)
            {
                CX[i] = 1;
                CX[i] = CX[i] + c[j] * mux(v, j);
            }
        }
        
        // A0=1; B0=1; D0=1; C0=1; 
        // // int[] memory c;
        // int[] memory c = new int[](features.length);
        // for (uint i =0; i < features.length; i++ )
        // {
        //     B0 = B0*v[i];
        //     c[i] = (features[i]-m[i])*(features[i]-m[i]);
        // }
        // for (uint i =0; i<c.length; i++){
        //     C0 = C0 + c[i]* mux(v, i); 
        // }
        // D0 = B0;
    }

    // function get_parameters() public view returns(int[],int[],int[],int[]){
    //     return(AX,BX,CX,DX);
    // }

    // save all the gasussian probability computations for all classes and computer probabilty values
    function cal_exponent(int e0, int e1, int f0, int f1, int g0, int g1, int h0, int h1, int Prior0,int Prior1) public 
    {
        A = e0*f1;
        B = f0*e1;
        C = (g0*h1)-(g1*h0);
        D = h0*h1;

        if (D != 0){
            Q = int(C/D);
            R = C%D;
        } else {
            Q = 0;
            R = 0;
        }
        
        
        edenR = 24*(D*D*D*D);
        enumR = 24*D*D*D*D+ R*24*D*D*D+R*R*12*D*D+R*R*R*4*D+R*R*R*R;
        if(Q>=0){
            Y0 = A*1*edenR*Prior0*Prior0;
            Y1 = B*1000*enumR*Prior1*Prior1;
        } else{
            Y0 = A*1000*edenR*Prior0*Prior0;
            Y1 = B*1*enumR*Prior1*Prior1;
        }
    }

    function get_exponent() public view returns (int, int){
        return (Y0,Y1);
    }

    // function compare()public view returns(int){
    //     // int p0=0; int p1=1;
    //     if(Y0 > Y1)return(0);
    //     else return(1);
    // }  

    function cal_probability(int Prior0, int Prior1, int Prior2, int Prior3, int Prior4, int Prior5) public
    {
        cal_exponent(AX[0],AX[1],BX[0],BX[1],CX[0],CX[1],DX[0],DX[1],Prior0, Prior1);
        if (Y0<Y1){P1=1000;P0=0;}else{P0=1000;P1=0;}
        cal_exponent(AX[2],AX[3],BX[2],BX[3],CX[2],CX[3],DX[2],DX[3],Prior2, Prior3);
        if (Y0<Y1){P3=1000;P2=0;}else{P2=1000;P3=0;}
        cal_exponent(AX[4],AX[5],BX[4],BX[5],CX[4],CX[5],DX[4],DX[5],Prior4, Prior5);
        if (Y0<Y1){P5=1000;P4=0;}else{P4=1000;P5=0;}
        if(P0>P1 && P2>P3 && P4>P5){        // P0,P2,P4{
            cal_exponent(AX[0],AX[2],BX[0],BX[2],CX[0],CX[2],DX[0],DX[2],Prior0, Prior2);
            if(Y0>Y1){P2=0;  //compare P0 and P4
                cal_exponent(AX[0],AX[4],BX[0],BX[4],CX[0],CX[4],DX[0],DX[4],Prior0, Prior4);
                P0=Y0; P4=Y1;}
            else{P0=0;
                cal_exponent(AX[2],AX[4],BX[2],BX[4],CX[2],CX[4],DX[2],DX[4],Prior2, Prior4);
                P2=Y0; P4=Y1;}
        }
        else if(P0>P1 && P2>P3 && P5>P4){   // P0,P2,P5
            cal_exponent(AX[0],AX[2],BX[0],BX[2],CX[0],CX[2],DX[0],DX[2],Prior0, Prior2);
            if(Y0>Y1){P2=0;  //compare P0 and P4
                cal_exponent(AX[0],AX[5],BX[0],BX[5],CX[0],CX[5],DX[0],DX[5],Prior0, Prior5);
                P0=Y0; P5=Y1;}
            else{P0=0;
                cal_exponent(AX[2],AX[5],BX[2],BX[5],CX[2],CX[5],DX[2],DX[5],Prior2, Prior5);
                P2=Y0; P5=Y1;}
        }
        else if(P0>P1 && P3>P2 && P4>P5)  { // P0,P3,P4
            cal_exponent(AX[0],AX[3],BX[0],BX[3],CX[0],CX[3],DX[0],DX[3],Prior0, Prior3);
            if(Y0>Y1){P3=0;  //compare P0 and P4
                cal_exponent(AX[0],AX[4],BX[0],BX[4],CX[0],CX[4],DX[0],DX[4],Prior0, Prior4);
                P0=Y0; P4=Y1;}
            else{P0=0;
                cal_exponent(AX[3],AX[4],BX[3],BX[4],CX[3],CX[4],DX[3],DX[4],Prior3, Prior4);
                P3=Y0; P4=Y1;}
        }
        else if(P0>P1 && P3>P2 && P5>P4){   // P0,P3,P5
            cal_exponent(AX[0],AX[3],BX[0],BX[3],CX[0],CX[3],DX[0],DX[3],Prior0, Prior3);
            if(Y0>Y1){P3=0;  //compare P0 and P4 <<<<<<<potential error: should be 0 vs 5
                cal_exponent(AX[0],AX[3],BX[0],BX[3],CX[0],CX[3],DX[0],DX[3],Prior0, Prior3);
                P0=Y0; P3=Y1;}  // <<<<<<<potential error: P5 = Y1
            else{P0=0;
                cal_exponent(AX[3],AX[5],BX[3],BX[5],CX[3],CX[5],DX[3],DX[5],Prior3, Prior5);
                P3=Y0; P5=Y1;}
        }
        else if(P1>P0 && P2>P3 && P4>P5){   // P1,P2,P4
            cal_exponent(AX[1],AX[2],BX[1],BX[2],CX[1],CX[2],DX[1],DX[2],Prior1, Prior2);
            if(Y0>Y1){P2=0;  //compare P1 and P4
                cal_exponent(AX[1],AX[4],BX[1],BX[4],CX[1],CX[4],DX[1],DX[4],Prior1, Prior4);
                P1=Y0; P4=Y1;}
            else{P1=0;
                cal_exponent(AX[2],AX[4],BX[2],BX[4],CX[2],CX[4],DX[2],DX[4],Prior2, Prior4);
                P2=Y0; P4=Y1;}
        }
        else if(P1>P0 && P2>P3 && P5>P4){   // P1,P2,P5
            cal_exponent(AX[1],AX[2],BX[1],BX[2],CX[1],CX[2],DX[1],DX[2],Prior1, Prior2);
            if(Y0>Y1){P2=0;  //compare P1 and P5
                cal_exponent(AX[1],AX[5],BX[1],BX[5],CX[1],CX[5],DX[1],DX[5],Prior1, Prior5);
                P1=Y0; P5=Y1;}
            else{P1=0;
                cal_exponent(AX[2],AX[5],BX[2],BX[5],CX[2],CX[5],DX[2],DX[5],Prior2, Prior5);
                P2=Y0; P5=Y1;}
        }
        else if(P1>P0 && P3>P2 && P4>P5){   // P1,P3,P4
            cal_exponent(AX[1],AX[3],BX[1],BX[3],CX[1],CX[3],DX[1],DX[3],Prior1, Prior3);
            if(Y0>Y1){P3=0;  //compare P1 and P4
                cal_exponent(AX[1],AX[4],BX[1],BX[4],CX[1],CX[4],DX[1],DX[4],Prior1, Prior4);
                P1=Y0; P4=Y1;}
            else{P1=0;
                cal_exponent(AX[3],AX[4],BX[3],BX[4],CX[3],CX[4],DX[3],DX[4],Prior3, Prior4);
                P3=Y0; P4=Y1;}
        }
        else{    // P1,P3,P5
            cal_exponent(AX[1],AX[3],BX[1],BX[3],CX[1],CX[3],DX[1],DX[3],Prior1, Prior3);
            if(Y0>Y1){P3=0;  //compare P1 and P5
                cal_exponent(AX[1],AX[5],BX[1],BX[5],CX[1],CX[5],DX[1],DX[5],Prior1, Prior5);
                P1=Y0; P5=Y1;}
            else{P1=0;
                cal_exponent(AX[3],AX[5],BX[3],BX[5],CX[3],CX[5],DX[3],DX[5],Prior3, Prior5);
                P3=Y0; P5=Y1;}
        }
    }
    function printall() public view returns(int,int,int,int,int,int){ 
        return(P0,P1,P2,P3,P4,P5); 
    }

    function predict()public view returns(int)
    {
        int p0=0; int p1=1; int p2=2; int p3=3; int p4=4;int p5=5;
        if(P0>P1 && P0>P2 && P0>P3 && P0>P4 && P0>P5)return(p0);
        else if (P1>P2 && P1>P0 && P1>P3 && P1>P4 && P1>P5)return(p1);
        else if (P2>P0 && P2>P1 && P2>P3 && P2>P4 && P2>P5)return(p2);
        else if (P3>P0 && P3>P1 && P3>P2 && P3>P4 && P3>P5)return(p3);
        else if (P4>P0 && P4>P1 && P4>P2 && P4>P3 && P4>P5)return(p4);
        else return(p5);
    }

}