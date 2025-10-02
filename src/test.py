from src.timeaverages import timeaverage
import matplotlib.pyplot as plt
import numpy as np
import torch 
import metatensor as mts
from metatensor import TensorBlock, TensorMap, Labels
from scipy.ndimage import gaussian_filter

maxlag=50
delta=np.zeros(maxlag)
delta[maxlag//2]=1
kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) 
print(np.sum(kernel))
plt.plot(kernel)
plt.show()
exit()


a = np.zeros(10)
a[5]=1
for i in range(10):
    print(np.roll(a,i))



exit()
M = 2
N = 10 
values = np.random.normal(0, 3, (M*N,1))
samples = np.array([[i,j] for i in np.arange(M) for j in np.arange(N)])
print(samples.shape)
print(values.shape)
print(values.dtype)
print(type(values))
block = TensorBlock(
            values=values,
            samples=Labels(["system", "atoms"], samples),
            components=[],
            properties=Labels("feat", np.array([[0]])),
            #properties=Labels("soap_pca", torch.tensor([[0], [1]])),
        )

tmap = TensorMap(keys=Labels("_", np.array([[0]])), blocks=[block])

print(tmap)
print('old', tmap.block().samples)

samples_selection = Labels(
        names=["system", "atoms"],
        values=samples[:4,:] # select first 4 samples
    )

feat1 = mts.split(
            tmap,
            axis="samples",
            selections=[
                samples_selection,
            ],
        )
print(samples_selection)
print(samples_selection.values.shape)
print('samplesel', feat1[0].block().samples)

exit()
atomsamples = np.array([[j] for j in np.arange(N)[:3]]) # select only the first 2 atoms
atoms_selection = Labels(
        names=["atoms"],
        values=atomsamples,
    )

feat2 = mts.split(
            tmap,
            axis="samples",
            selections=[
                atoms_selection,
            ],
        )

print('atomsel', feat2[0].block().samples)