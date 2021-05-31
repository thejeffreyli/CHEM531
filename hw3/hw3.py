import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as integrate
import math


# known values
e = 0.00054858
u = 1.00784/(2*e)
w = 0.01822
conv = 219474.63 #conversion for hartree to wavenumbers

# harmonic oscillator
def idealEnergy(v):
    return (v + 0.5)*w *conv

# basic functions
def pibBasis(x, n, L):
    return math.sqrt(2/L) * math.sin(n * math.pi * (x + L/2) / L)

def matMaker(L, size):
    matrixIn = np.zeros((size,size), dtype = float) #Intialize sizexsize matrix
    # Note: undefined at 0
    for m in range(1, size+1):
        for n in range(1, size+1): 
            matElem = matrix(m, n, L) #create Matrix element object
            if m==n: #if m==n, then H= <n|T|n> + <m|V|n>
                matrixIn[m - 1, n - 1]= matElem.tMat() + matElem.vMat()[0] 
            else: #if m!=n then H= <m|V|n> 
                matrixIn[m -1 ,n - 1]= matElem.vMat()[0]
    return matrixIn

def diagSorted(matrix, size): #diagonlizes and sorts eigenvalues
    #diagonalize matrix             
    eigVal, eigVec = la.eigh(matrix) #extract the eigenvalues, eigenvectors
    eigVal = eigVal.real * conv
    return eigVal, eigVec

class matrix(): #creates H matrix
    
    # self represents the instance of the class
    def __init__(self, m, n, L):
        self.m = m
        self.n = n
        self.L = L
        
    # wavefunct is represetned by Sqrt[2/L]*Sin[(m*\[Pi]*(x + L/2)/L)]
    def mWav(self, x): # m wavefunction (given)
        return math.sqrt(2/self.L) * math.sin(self.m * math.pi * (x + self.L/2) / self.L)
    
    def nWav(self, x): # n wavefunction (given)
        return math.sqrt(2/self.L) * math.sin(self.n * math.pi * (x + self.L/2) / self.L)
    
    def tMat(self): #kinetic matrix: ((n^2) (\[Pi]^2))/(2*u*L^2) 
        return ((self.n**2) * (math.pi**2)) / (2*u*self.L**2)  

    def vOper(self, x): #operator
        return 0.5 * u * w**2 * x**2

    def vMat (self): #potential matrix
        setup = lambda x: self.mWav(x) * self.vOper(x) * self.nWav(x)
        return integrate.quad(setup, -self.L/2, self.L/2)  
    

def main():
    
    'PROBLEM ONE'
    
    'Pick a “reasonable” size H-matrix (at least 20x20) and investigate 5 or so'
    'values of L to get the first six eigenvalues. (Use the classical turning'
    'points for v = 5 to get the minimum value of L) and plot the results vs L'
    'for v up to 5. Comment on the results and indicate the exact energies'
    'on the plot.'
    
    
    # find ideal energies through HO: Ev = hw(v + 1/2), v = 0 to infinity
    ideal = []
    for v in range (0,6):
        ideal.append(idealEnergy(v))    
    ideal = np.array(ideal)
    print("First Six Ideal Energies: " , ideal)
    
    size = 20
    L = 3.25 # optimal length of box
    
    m1 = matMaker(3.25, size)    
    eigVal, eigVec = diagSorted(m1, size)
    
    eigVecArr = []
    for i in range(0,6):
        eigVecArr.append(eigVal[i])
        
    # compare the eiganvalues with the acceped ideal energies    
    print("First Six Eigenvalues: ", eigVecArr)


    'PROBLEM TWO'
    
    'Using the “optimum” L determine the eigenfunctions and plot them for'
    'v up to 5. Comment on the number of oscillations and also the'
    'symmetry of them.'

    xeval = np.linspace(-L/2, L/2, 10001)
    eigfunc = np.zeros((xeval.shape[0]))
    
    # order from v  = 0 to v = 5 
    order = 5

    # determine eigenfunctions
    for n in range(20):
        for j in range(xeval.shape[0]):
            eigfunc[j] += np.sum(eigVec[n, order] * pibBasis(xeval[j], n+1, L))
    #print (eigfunc)

    # plotting
    fig, ax1 = plt.subplots() 
    normalization = integrate.simps((eigfunc)**2,xeval)
    normalpsi=(np.sqrt(conv)*eigfunc/np.sqrt(normalization))
    # plotting eigenfunction
    ax1.plot(xeval, normalpsi + idealEnergy(order), color='blue', label = 'eigenfunction')    
    ax1.axvline(x= -1*L/2, color='gray', linestyle='dashed')
    ax1.axvline(x= L/2, color='gray', linestyle='dashed')
    ax1.axvline(x=0, color='gray', linestyle='dashed')
    ax1.axis([-L/2 - 0.1, L/2 + 0.1, -(1.1*np.max(normalpsi)) + idealEnergy(order), 
              1.1*np.max(normalpsi) + idealEnergy(order)])
    ax1.axhline(y = idealEnergy(order), color='orange', linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('energy cm-1')
    plt.title('Plot, v = 5 ')

    


if __name__ == "__main__":
    main()   