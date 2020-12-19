import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as la



# Parameters, all in Atomic Units
a = 0.060503 *219474 
b = 0.008453 *219474 
c = 1.1954
v0 = 0.05624 *219474 
w0 = 0.015001
# alpha = 6
# n = 2
mass = 4665.664

def diagSorted(matrix, size): #diagonlizes and sorts eigenvalues
    #diagonalize matrix             
    eigVal, eigVec = la.eigh(matrix) #extract the eigenvalues, eigenvectors
    #print(eigVec)
    eigVal = eigVal.real #real eigenvalues
    diagMat = np.diag(eigVal) #diagonal matrix 
    #diagMat = diagMat * 219474
    #print(f" eigenvalues in diagonal matrix \n {diagMat}")
    
    #getting the exact energy levels
    list = [] 
    for i in range(size):
        list.append(diagMat[i][i])  
    #sorted order
    list = sorted(list)
    
    return eigVal, eigVec, list

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
    matrixIn = np.round(matrixIn,4) #round
    return matrixIn 

class matrix(): #creates H matrix

    # self represents the instance of the class
    def __init__(self, m, n, L):
        self.m = m
        self.n = n
        self.L = L
        
    # pib basis
    def mWav(self, x): # m wavefunction (given)
        return np.sqrt(2/self.L) * np.sin(self.m * np.pi * (x + self.L/2) / self.L)
    
    def nWav(self, x): # n wavefunction (given)
        return np.sqrt(2/self.L) * np.sin(self.n * np.pi * (x + self.L/2) / self.L)
    
    def veff(self, x):
        return 0.5*a*(x**2) + 0.5*b*(x**4) + v0*np.exp(-c*(x**2)) + (self.n + 0.5) * w0 
    
    def vMat(self):
        setupV = lambda x: self.mWav(x) * self.veff(x) * self.nWav(x)
        return integrate.quad(setupV, -self.L/2, self.L/2)

    def tMat(self):
        return ((self.n**2) * (np.pi**2)) / (2*mass*self.L**2)  
    
    
    
def main():
    
    "PROBLEM 1 A"
    
    size = 10
    matrix1 = matMaker(3, size)
    eigVal, eigVec, sortedList = diagSorted(matrix1, size)
    print (eigVal)

    "PROBLEM 1 C"

    L = 3    
    Eht = 219474
    
    #PIB Basis function
    def pib(x, n, L):
        psi = np.sqrt(2./L) * np.sin(((float(n)*x/L) + (float(n)/2.))*np.pi)
        return psi

    # testarr = np.arange(6)
    # def hoenergy(omega,v):
    #     return omega*((v)+.5)
    # test = hoenergy(w0, testarr) * 219474 
    # print(test)


    xeval = np.linspace(-L/2, L/2, 10001)
    eigfunc = np.zeros((xeval.shape[0]))
    order = 0 # indicates which quantum number
    
    for n in range(10):
        for j in range(xeval.shape[0]):
            eigfunc[j] += np.sum(eigVec[n, order] * pib(xeval[j], n+1, L))
    # print (eigfunc)
    
    
    fig, ax1 = plt.subplots() 
    x = np.linspace(-1.6, 1.6, 100)    
    # y1 = 0.5*a*(x**2) + 0.5*b*(x**4) + v0*np.exp(-c*(x**2)) 


    # plt.plot(x,y1, 'black')    
    # normalized eigenfunction
    normalization = integrate.simps((eigfunc)**2, xeval)
    normalpsi=(np.sqrt(Eht)*eigfunc/np.sqrt(normalization))
    
    # box length
    ax1.axvline(x=-1*L/2, color='red', linestyle='dashed')
    ax1.axvline(x=L/2, color='red', linestyle='dashed')
    
    ax1.plot(xeval, (normalpsi+10480)*10**(-4))
    # plt.ylim(1, 1.6)
    # ax1.plot(poteval,Eht*(potential(poteval,k)),label='Potential')
    ax1.axvline(x=0, color='k', linestyle='dashed')
    plt.xlabel('X')
    plt.ylabel('V_x * 10^-4 (cm^-1)')
    plt.title('Wavefunction, First Eigenvalue (0)')    
if __name__=="__main__":
   main()
   

   
   
