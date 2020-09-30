# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:12:33 2020

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as la

class matrix():
      
    #known values
    e = 0.00054858
    u = 1.00784/(2*e)
    w = 0.01822    
    
    # self represents the instance of the class
    def __init__(self, m, n, L, re, de, a):
        self.m = m
        self.n = n
        self.L = L
        self.re = re
        self.de = de
        self.a = a

    #PIB Wavefunctions 
    def mWav(self, r): # m wavefunction (given)
        return np.sqrt(2/self.L) * np.sin(self.m * np.pi * (r - self.re + self.L/2) / self.L)
    
    def nWav(self, r): # n wavefunction (given)
        return np.sqrt(2/self.L) * np.sin(self.n * np.pi * (r - self.re + self.L/2) / self.L)

    #morse potential
    def vM(self, r):
        return self.de * (1- np.exp(-self.a * (r - self.re)))**2  
    
    #from hw2: kinetic matrix
    def tMat(self): #kinetic matrix: ((n^2) (\[Pi]^2))/(2*u*L^2) 
        return ((self.n**2) * (np.pi**2)) / (2 * matrix.u * self.L**2) 
    
    #potential matrix
    def vMat (self):
        setupV = lambda r: self.mWav(r) * self.vM(r) * self.nWav(r)
        #adjusted to re
        return integrate.quad(setupV, self.re - self.L/2, self.re + self.L/2)
    

    
def main():
    
    
    'PROBLEM ONE:' 
    'The Morse Potential (energies and wavefunctions).' 
    'Plot this for two cases (use the same frame of x and y.'
    
    'For case a) use a PIB basis (box 0 to L) to obtain the first 5 energy eigenvalues'
    'and wavefunctions converged to at least three significant figures. Use the reduced'
    'mass of the H2 molecule, which is mp/2.'
        
   
    #values for A
    de = 4.7/27.211 #hartrees
    a = 1.0 
    re= 1.4 
    L = 2.4
  
    matrixIn = np.zeros((50,50), dtype = float) #Intialize 50x50 matrix
    # Note: undefined at 0
    for m in range(1,51):
        for n in range(1,51): 
            matElem = matrix(m, n, L, re, de, a) #create Matrix element object
            if m==n: #if m==n, then H= <n|T|n> + <m|V|n>
                matrixIn[m - 1, n - 1]= matElem.tMat()+ matElem.vMat()[0] 
            else: #if m!=n then H= <m|V|n> 
                matrixIn[m -1 ,n - 1]= matElem.vMat()[0]    
    matrixIn = np.round(matrixIn,4) #round     
    #print(matrixIn)   
    
    #diagonalize matrix             
    eigVal, eigVec = la.eigh(matrixIn) #extract the eigenvalues, eigenvectors
    #print(eigVec)
    eigVal= eigVal.real #real eigenvalues
    diagMat= np.diag(eigVal) #diagonal matrix 
    diagMat = diagMat * 219474
    #print(f" eigenvalues in diagonal matrix \n {diagMat}")
             
    #getting the exact energy levels
    list = [] 
    for i in range(0,50):
        list.append (diagMat[i][i])  
    #sorted order
    list = sorted(list)
    #print(list)

    #prints first 5 eigenvalues
    sortedList = []
    for i in range(0,5):
        sortedList.append(list[i])
    print(f" eigenvalues in diagonal matrix, cm^-1\n {sortedList}")




    xeval = np.linspace(re - L/2, re + L/2, 10001)
    pib = np.zeros((len(xeval), 50))
    for n in range(1, 51):
        pibElem = matrix(n, n, L, re, de, a)
        pib[:, n - 1] = pibElem.nWav(xeval)
  
    
    wavevalues = pib @ eigVec
    x = np.arange(0, 20, 0.1)
    ax1 = plt.subplot(111)
    ax1.plot(x, matElem.vM(x))
    ax1.vlines(re - L / 2, 0, 2, colors="orange", linestyles="dashed")
    ax1.vlines(re + L / 2, 0, 2, colors="orange", linestyles="dashed")          
    plt.xlim(0, 3.25)
    plt.ylim(-0.01, 0.3)
    ax1.set_aspect("auto")
    
    for n in range(0, 5):
        ax1.plot(xeval, wavevalues[:, n] / 200 + sortedList[n]/219474) 
    ax1.grid(True)
    plt.xlabel('r, bohr')
    plt.ylabel('energy, cm-1')
    plt.title('Morse Potential (A, j = 0)')
    plt.show()


          
if __name__ == "__main__":
    main()
