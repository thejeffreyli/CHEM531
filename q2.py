import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as la


"PROBLEM TWO"

H2WNx = 219474 * (10**(-4))

# Parameters, all in Atomic Units
a = 0.060503 
b = 0.008453
c = 1.1954
v0 = 0.05624 
w0 = 0.015001
# alpha = 6
# n = 2
mass = 4665.664

L = 3.

def veff(n, x):
    return 0.5*a*(x**2) + 0.5*b*(x**4) + v0*np.exp(-c*(x**2)) + (n + 0.5) * w0

xeval = np.linspace(-L/2, L/2, 10001)

y1 = veff(0, xeval) * H2WNx
y2 = veff(1, xeval) * H2WNx
y3 = veff(2, xeval) * H2WNx



fig, ax1 = plt.subplots()
ax1.plot(xeval, y1)


# ax1.axvline(x=0, color='red', linestyle='dashed')
# ax1.axhline(y= 1.3989382497000002, color='red', linestyle='dashed')

# plt.ylim(1, 1.8)
# plt.xlim(-1.6, 1.6)
# plt.xlabel('X')
# plt.ylabel('Veff * 10^-4 (cm^-1)')
# plt.title('Effective Adiabatic Potential, n_y =0')  

fig, ax1 = plt.subplots()
ax1.plot(xeval, y2)



# ax1.axvline(x=0, color='red', linestyle='dashed')
# ax1.axhline(y= 1.7281711971, color='red', linestyle='dashed')

# plt.ylim(1.4, 2.2)
# plt.xlim(-1.6, 1.6)
# plt.xlabel('X')
# plt.ylabel('Veff * 10^-4 (cm^-1)')
# plt.title('Effective Adiabatic Potential, n_y =1')



fig, ax1 = plt.subplots()
ax1.plot(xeval, y3)

ax1.axvline(x=0, color='red', linestyle='dashed')
ax1.axhline(y= 2.0574041445000004, color='red', linestyle='dashed')

plt.ylim(1.8, 2.6)
plt.xlim(-1.6, 1.6)
plt.xlabel('X')
plt.ylabel('Veff * 10^-4 (cm^-1)')
plt.title('Effective Adiabatic Potential, n_y =2')
   
   
   


