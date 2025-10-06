import numpy as np

def fourier_trafo(X):
    # let X be of shape T,N,P
    # run a realFFT in the first dimension to see the autocorrelation....

    FT = np.fft.rfft(X, axis=0)

    return FT