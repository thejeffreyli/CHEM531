# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:18:07 2020

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as integrate
import math
from sympy.solvers import solve
from sympy import Symbol

class matrix(): #creates H matrix
    
    #known values
    e = 0.00054858
    u = 1.00784/(2*e)
    w = 0.01822
    
    # self represents the instance of the class
    def __init__(self, m, n, L):
        self.m = m
        self.n = n
        self.L = L
        
    #Sqrt[2/L]*Sin[(m*\[Pi]*(x + L/2)/L)]
    def mWav(self, x): # m wavefunction (given)
        return math.sqrt(2/self.L) * math.sin(self.m * math.pi * (x + self.L/2) / self.L)
    
    def nWav(self, x): # n wavefunction (given)
        return math.sqrt(2/self.L) * math.sin(self.n * math.pi * (x + self.L/2) / self.L)
    
    def tMat(self): #kinetic matrix: ((n^2) (\[Pi]^2))/(2*u*L^2) 
        return ((self.n**2) * (math.pi**2)) / (2*matrix.u*self.L**2)  

    def vOper(self, x): #operator
        return 0.5 * matrix.u * matrix.w**2 * x**2

    def vMat (self): #potential matrix
        setup = lambda x: self.mWav(x) * self.vOper(x) * self.nWav(x)
        return integrate.quad(setup, -self.L/2, self.L/2)  
    
    def vEff (self): #potential matrix
        setup = lambda x: self.mWav(x) * self.vOper(x) * self.nWav(x)
        return integrate.quad(setup, -self.L/2, self.L/2)  
    

def main():
    
    #PROBLEM ONE: INVESTIGATE 20x20 matrix and find best L value
    
    matrixIn = np.zeros((20,20), dtype = float) #Intialize 20x20 matrix
    for m in range(0,20):
        for n in range(0,20): 
            matElem = matrix(m,n,4) #create Matrix element object
            
            if m==n: #if m==n, then H= <n|T|n> + <m|V|n>
                matrixIn[m,n]= matElem.tMat()+ matElem.vMat()[0] 
                
            else: #if m!=n then H= <m|V|n> 
                matrixIn[m,n]= matElem.vMat()[0]
                
    #matrixIn = np.round(matrixIn,4)               
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
    for i in range(0,20):
        list.append (diagMat[i][i])  
    
    #sorted order
    list = sorted(list)
    print(list)

    
    #prints in order
    for i in range(1,7):
        print(list[i])
        
    ideal = []
    for v in range (1,7):
        ideal.append(1*matrix.w*(v+0.5)*219474)    
    #print (ideal)

    
    def Ev(v): #f(x)
        return (v+0.5)*matrix.w*219474
    
    def HO(x): #(x)
        return 0.5*matrix.u*(matrix.w**2)*(x**2)*219474  
    
    
    #plot boundaries
    d = 0.9
    x = np.linspace(-d, d)
    plt.plot(x, HO(x))
    for i in range(6):
        plt.hlines(Ev(i),-d, d)
    plt.title('Plot, v = 5')
    plt.xlabel("distance (bohr)")
    plt.ylabel("wavenumbers (cm^-1)")
    
    plt.axhline(y = 21997.453368028, color='r', linestyle='-')
    
    #classical turning point, solve the interception between 
    #the straight line and the HO
    x = Symbol('x')
    x = solve(0.5*matrix.u*(matrix.w**2)*(x**2)*219474 -  21997.453368021, x)
    print(x)
    
    plt.show()



#PROBLEM TWO: Determine the eigenfunctions
#optimum L = 3.25
    L = 3.25    
    Eht = 219474

    #PIB Basis function
    def pib(x, n, L):
        psi = np.sqrt(2./L) * np.sin(((float(n)*x/L) + (float(n)/2.))*np.pi)
        return psi

    def hoenergy(omega,v):
        return omega*(float(v)+.5)


    
    #linspace: Return evenly spaced numbers over a specified interval
    xeval = np.linspace(-L/2, L/2, 10001)
    eigfunc = np.zeros((xeval.shape[0]))
    order = 5
    omegaht = 0.01822

    
    for n in range(20):
        for j in range(xeval.shape[0]):
            eigfunc[j] += np.sum(eigVec[n, order] * pib(xeval[j], n+1, L))
    #print (eigfunc)

    fig, ax1 = plt.subplots() #matplotlib functions
    normalization=integrate.simps((eigfunc)**2,xeval)
    normalpsi=(np.sqrt(Eht)*eigfunc/np.sqrt(normalization))
    ax1.axvline(x=-1*L/2, color='b', linestyle='dashed')
    ax1.axvline(x=L/2, color='b', linestyle='dashed',label='L/2')
    
    
    ax1.plot(xeval,normalpsi+Eht*hoenergy(omegaht,order),label='eigenfunctuon')
    # ax1.plot(poteval,Eht*(potential(poteval,k)),label='Potential')
    ax1.axvline(x=0, color='k', linestyle='dashed')
    ax1.axis([-L/2-.1,L/2+.1,-(1.1*np.max(normalpsi))+Eht*hoenergy(omegaht,order),1.1*np.max(normalpsi)+Eht*hoenergy(omegaht,order)])
    ax1.axhline(y=Eht*hoenergy(omegaht,order), color='k', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('energy cm-1')
    plt.title('Plot, v = 5 ')

    



if __name__ == "__main__":
    main()    