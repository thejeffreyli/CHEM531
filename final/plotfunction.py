import matplotlib.pyplot as plt
import numpy as np


"PROBLEM 1 B"

a = 0.060503  *219474 *10**(-4)
b = 0.008453 *219474  *10**(-4)
c = 1.1954
v0 = 0.05624 *219474  *10**(-4)
w0 = 0.015001
# alpha = 6
# n = 2
mass = 4665.664

# 100 linearly spaced numbers
x = np.linspace(-1.6, 1.6, 100)

# the function, which is y = x^2 here
# n = np.arange(5)
y1 = 0.5*a*(x**2) + 0.5*b*(x**4) + v0*np.exp(-c*(x**2)) 



# setting the axes at the centre
fig = plt.figure()


plt.plot(x,y1, 'black')
plt.axhline(y=1.10505278, color='blue', linestyle='-')
plt.axhline(y=1.10766791, color='green', linestyle='-')
plt.axhline(y=1.19579544, color='orange', linestyle='-')
plt.axhline(y=1.19964038, color='magenta', linestyle='-')
plt.axhline(y=1.2571796, color='purple', linestyle='-')
plt.axhline(y=1.27657634, color='yellow', linestyle='-')




plt.ylim(1, 1.6)
plt.title("Adiabatic Approximation of Figure 1")
plt.xlabel('X')
plt.ylabel('V_x * 10^-4 (cm^-1)')
# [1.12186085 1.12292204 1.23491096 1.2495012  1.30145697 1.34287861
#  1.7187569  1.84831808 2.78041431 2.89088006]
#1.10505278 1.10766791 1.19579544 1.19964038 1.25717965 1.27657634
# show the plot
plt.show()
