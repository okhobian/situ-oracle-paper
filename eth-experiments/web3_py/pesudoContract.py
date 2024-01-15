import numpy as np
import random

class PredictSituBayes:
    
    num_situ = 6
    P0 = P1 = P2 = P3 = P4 = P5 = 0
    AX = BX = CX = DX = np.zeros(num_situ, dtype=int)
    A = B = C = D = Q = R = edenR = enumR = 0
    Y0 = Y1 = 0
    
    def __init__(self):
        pass
    
    def mux(self, v, k):
        temp = 1
        for i in range(len(v)):
            if(i==k): continue
            else:  temp = temp * v[i]
        return temp
    
    def cal_parameters(self, features, v, m):
    
        for i in range(self.num_situ):
            self.AX[i] = 1 
            c = np.zeros(len(features), dtype=int)
            
            for j in range(len(features)):
                self.BX[i] = 1
                self.DX[i] = 1        
                
                # print("before:", self.BX[i])        
                self.BX[i] = self.BX[i] * v[j]
                self.DX[i] = self.DX[i] * v[j]
                # print("after:", self.BX[i])   
                # print("===")
                c[j] = (features[j]-m[j])*(features[j]-m[j])
            
            for j in range(len(features)):
                self.CX[i] += c[j] * self.mux(v, j)
    
    def cal_exponent(self, e0, e1, f0, f1, g0, g1, h0, h1, Prior0, Prior1):
        self.A = e0 * f1
        self.B = f0 * e1
        self.C = (g0*h1) - (g1*h0)
        self.D = h0 * h1
        
        if not self.D == 0:
            self.Q = int(self.C / self.D)
            self.R = self.C % self.D
                
        self.edenR = 24*(self.D**4)
        # enumR = (24*self.D**4) + (24*self.R*self.D**3) + (12*self.R**2*self.D**2)+ (4*self.R**3*self.D+self.R**4)
        self.enumR = 24*(self.D**4) + (24*self.R*(self.D**3)) + (12*(self.R**2)*(self.D**2))+ (4*(self.R**3)*self.D+(self.R**4))
        
        if self.Q>=0:
            self.Y0 = self.A * 1 * self.edenR * Prior0 * Prior0
            self.Y1 = self.B * 1000 * self.enumR * Prior1 * Prior1
        else:
            self.Y0 = self.A * 1000 * self.edenR * Prior0 * Prior0
            self.Y1 = self.B * 1 * self.enumR * Prior1 * Prior1
            
    def cal_probability(self, Prior0, Prior1, Prior2, Prior3, Prior4, Prior5):
    
        # print("AX", self.AX)
        # print("BX", self.BX)
        # print("CX", self.CX)
        # print("DX", self.DX)
    
        ## PO vs. P1
        self.cal_exponent(self.AX[0],self.AX[1],self.BX[0],self.BX[1],self.CX[0],self.CX[1],self.DX[0],self.DX[1], Prior0, Prior1)
        # print("0v1", self.Y0, self.Y1)
        if self.Y0 < self.Y1:
            self.P1=1000
            self.P0=0
        else: 
            self.P0=1000
            self.P1=0
            
        ## P2 vs. P3
        self.cal_exponent(self.AX[2],self.AX[3],self.BX[2],self.BX[3],self.CX[2],self.CX[3],self.DX[2],self.DX[3], Prior2, Prior3)
        # print("2v3", self.Y0, self.Y1)
        if self.Y0 < self.Y1:
            self.P3=1000
            self.P2=0
        else:
            self.P2=1000
            self.P3=0
            
        ## P4 vs. P5
        self.cal_exponent(self.AX[4],self.AX[5],self.BX[4],self.BX[5],self.CX[4],self.CX[5],self.DX[4],self.DX[5], Prior4, Prior5)
        # print("4v5", self.Y0, self.Y1)
        if self.Y0 < self.Y1:
            self.P5=1000
            self.P4=0
        else:
            self.P4=1000
            self.P5=0
        
        # print(">>>", self.P0, self.P1, self.P2, self.P3, self.P4, self.P5, self.Y0, self.Y1)
        
        ## P0, P2, P4 wins previous round
        if (self.P0 > self.P1 and self.P2 > self.P3 and self.P4 > self.P5):
            
            ## P0 vs. P2
            self.cal_exponent(self.AX[0],self.AX[2],self.BX[0],self.BX[2],self.CX[0],self.CX[2],self.DX[0],self.DX[2], Prior0, Prior2)
            if self.Y0 > self.Y1:
                self.P2=0 
                
                ## P0 vs. P4
                self.cal_exponent(self.AX[0],self.AX[4],self.BX[0],self.BX[4],self.CX[0],self.CX[4],self.DX[0],self.DX[4], Prior0, Prior4)
                self.P0=self.Y0
                self.P4=self.Y1
                
            else:
                self.P0=0
                
                ## P2 vs. P4
                self.cal_exponent(self.AX[2],self.AX[4],self.BX[2],self.BX[4],self.CX[2],self.CX[4],self.DX[2],self.DX[4], Prior2, Prior4)
                self.P2=self.Y0
                self.P4=self.Y1
        
        ## P0, P2, P5 wins previous round
        elif (self.P0 > self.P1 and self.P2 > self.P3 and self.P5 > self.P4):
            
            ## P0 vs. P2
            self.cal_exponent(self.AX[0],self.AX[2],self.BX[0],self.BX[2],self.CX[0],self.CX[2],self.DX[0],self.DX[2], Prior0, Prior2)
            if self.Y0 > self.Y1:
                self.P2=0
                
                ## P0 vs. P5
                self.cal_exponent(self.AX[0],self.AX[5],self.BX[0],self.BX[5],self.CX[0],self.CX[5],self.DX[0],self.DX[5], Prior0, Prior5)
                self.P0=self.Y0
                self.P5=self.Y1
                
            else:
                self.P0=0
                
                ## P2 vs. P5
                self.cal_exponent(self.AX[2],self.AX[5],self.BX[2],self.BX[5],self.CX[2],self.CX[5],self.DX[2],self.DX[5], Prior2, Prior5)
                self.P2=self.Y0
                self.P5=self.Y1
        
        ## P0, P3, P4 wins previous round
        elif (self.P0 > self.P1 and self.P3 > self.P2 and self.P4 > self.P5):
            
            ## P0 vs. P3
            self.cal_exponent(self.AX[0],self.AX[3],self.BX[0],self.BX[3],self.CX[0],self.CX[3],self.DX[0],self.DX[3], Prior0, Prior3)
            if self.Y0 > self.Y1:
                self.P3=0  
                
                ## P0 vs. P4
                self.cal_exponent(self.AX[0],self.AX[4],self.BX[0],self.BX[4],self.CX[0],self.CX[4],self.DX[0],self.DX[4], Prior0, Prior4)
                self.P0=self.Y0
                self.P4=self.Y1
                
            else:
                self.P0=0
                
                ## P3 vs. P4
                self.cal_exponent(self.AX[3],self.AX[4],self.BX[3],self.BX[4],self.CX[3],self.CX[4],self.DX[3],self.DX[4], Prior3, Prior4)
                self.P3=self.Y0
                self.P4=self.Y1
        
        ## P0, P3, P5 wins previous round
        elif (self.P0 > self.P1 and self.P3 > self.P2 and self.P5 > self.P4):
            
            ## P0 vs. P3
            self.cal_exponent(self.AX[0],self.AX[3],self.BX[0],self.BX[3],self.CX[0],self.CX[3],self.DX[0],self.DX[3], Prior0, Prior3)
            if self.Y0 > self.Y1:
                self.P3=0  
                
                ## P0 vs. P5 #### <<<< potential error
                self.cal_exponent(self.AX[0],self.AX[5],self.BX[0],self.BX[5],self.CX[0],self.CX[5],self.DX[0],self.DX[5], Prior0, Prior5)
                self.P0=self.Y0
                self.P5=self.Y1
                
            else:
                self.P0=0
                
                ## P3 vs. P5
                self.cal_exponent(self.AX[3],self.AX[5],self.BX[3],self.BX[5],self.CX[3],self.CX[5],self.DX[3],self.DX[5], Prior3, Prior5)
                self.P3=self.Y0
                self.P5=self.Y1
        
        ## P1, P2, P4 wins previous round
        elif (self.P1 > self.P0 and self.P2 > self.P3 and self.P4 > self.P5):
            
            ## P1 vs. P2
            self.cal_exponent(self.AX[1],self.AX[2],self.BX[1],self.BX[2],self.CX[1],self.CX[2],self.DX[1],self.DX[2], Prior1, Prior2)
            if self.Y0 > self.Y1:
                self.P2=0  
                
                ## P1 vs. P4
                self.cal_exponent(self.AX[1],self.AX[4],self.BX[1],self.BX[4],self.CX[1],self.CX[4],self.DX[1],self.DX[4], Prior1, Prior4)
                self.P1=self.Y0
                self.P4=self.Y1
                
            else:
                self.P1=0
                
                ## P2 vs. P4
                self.cal_exponent(self.AX[2],self.AX[4],self.BX[2],self.BX[4],self.CX[2],self.CX[4],self.DX[2],self.DX[4], Prior2, Prior4)
                self.P2=self.Y0
                self.P4=self.Y1
        
        ## P1, P2, P5 wins previous round
        elif (self.P1 > self.P0 and self.P2 > self.P3 and self.P5 > self.P4):
            
            # P1 vs. P2
            self.cal_exponent(self.AX[1],self.AX[2],self.BX[1],self.BX[2],self.CX[1],self.CX[2],self.DX[1],self.DX[2], Prior1, Prior2)
            if self.Y0 > self.Y1:
                self.P2=0
                
                ## P1 vs. P5
                self.cal_exponent(self.AX[1],self.AX[5],self.BX[1],self.BX[5],self.CX[1],self.CX[5],self.DX[1],self.DX[5], Prior1, Prior5)
                self.P1=self.Y0
                self.P5=self.Y1
                
            else:
                self.P1=0
                
                ## P2 vs. P5
                self.cal_exponent(self.AX[2],self.AX[5],self.BX[2],self.BX[5],self.CX[2],self.CX[5],self.DX[2],self.DX[5], Prior2, Prior5)
                self.P2=self.Y0
                self.P5=self.Y1
        
        ## P1, P3, P4 wins previous round
        elif (self.P1 > self.P0 and self.P3 > self.P2 and self.P4 > self.P5):
            
            ## P1 vs. P3   
            self.cal_exponent(self.AX[1],self.AX[3],self.BX[1],self.BX[3],self.CX[1],self.CX[3],self.DX[1],self.DX[3], Prior1, Prior3)
            if self.Y0 > self.Y1:
                self.P3=0
                
                ## P1 vs. P4
                self.cal_exponent(self.AX[1],self.AX[4],self.BX[1],self.BX[4],self.CX[1],self.CX[4],self.DX[1],self.DX[4], Prior1, Prior4)
                self.P1=self.Y0
                self.P4=self.Y1
                
            else:
                self.P1=0
                
                ## P3 vs. P4
                self.cal_exponent(self.AX[3],self.AX[4],self.BX[3],self.BX[4],self.CX[3],self.CX[4],self.DX[3],self.DX[4], Prior3, Prior4)
                self.P3=self.Y0
                self.P4=self.Y1
        
        ## P1, P3, P5 wins previous round
        else:
            
            ## P1 vs. P3
            self.cal_exponent(self.AX[1],self.AX[3],self.BX[1],self.BX[3],self.CX[1],self.CX[3],self.DX[1],self.DX[3], Prior1, Prior3)
            if self.Y0 > self.Y1:
                self.P3=0
                
                ## P1 vs. P5
                self.cal_exponent(self.AX[1],self.AX[5],self.BX[1],self.BX[5],self.CX[1],self.CX[5],self.DX[1],self.DX[5], Prior1, Prior5)
                self.P1=self.Y0
                self.P5=self.Y1
                
            else:
                self.P1=0
                
                ## P3 vs. P5
                self.cal_exponent(self.AX[3],self.AX[5],self.BX[3],self.BX[5],self.CX[3],self.CX[5],self.DX[3],self.DX[5], Prior3, Prior5)
                self.P3=self.Y0
                self.P5=self.Y1
        
    def predict(self):
        
        # print(self.P0, self.P1, self.P2, self.P3, self.P4, self.P5)
        if(self.P0 > self.P1 and self.P0 > self.P2 and self.P0 > self.P3 and self.P0 > self.P4 and self.P0 > self.P5):
            return 0
        elif(self.P1 > self.P0 and self.P1 > self.P2 and self.P1 > self.P3 and self.P1 > self.P4 and self.P1 > self.P5):
            return 1
        elif(self.P2 > self.P0 and self.P2 > self.P1 and self.P2 > self.P3 and self.P2 > self.P4 and self.P2 > self.P5):
            return 2
        elif(self.P3 > self.P0 and self.P3 > self.P1 and self.P3 > self.P2 and self.P3 > self.P4 and self.P3 > self.P5):
            return 3
        elif(self.P4 > self.P0 and self.P4 > self.P1 and self.P4 > self.P2 and self.P4 > self.P3 and self.P4 > self.P5):
            return 4
        elif(self.P5 > self.P0 and self.P5 > self.P1 and self.P5 > self.P2 and self.P5 > self.P3 and self.P5 > self.P4):
            return 5
        return random.randint(0,6)
        