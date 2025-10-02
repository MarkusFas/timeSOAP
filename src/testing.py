import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(0,np.pi*4, 100)
x = np.linspace(0,np.pi*4, 100)
A = np.sin(x) + 2*np.cos(3*x) + np.sin(0.5*x) + np.cos(2*x)

B = np.fft.fft(A, axis=0)

plt.plot(A)
plt.show()

plt.plot(B)
plt.xlim(0,20)
plt.show()