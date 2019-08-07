import numpy as np
import scipy.signal as sc


def logMagStft(numpyArray, sample_rate, n_fft):
    """
    Calculates the logarithmic spectral map
    """
    f, t, sx = sc.stft(numpyArray, fs=sample_rate, nperseg=n_fft, noverlap=n_fft//2) 
    return np.log(np.abs(sx)+np.e**-10)


def ffts(ts):
    """
    calculates the absolut value of the fourier transform
    """
    cffarr=np.fft.rfft(ts)
    rffarr= np.empty(len(ts))
    for i in range(0, len(ts)//2):
        rffarr[2*i]=cffarr[i].real
        rffarr[2*i+1]=cffarr[i].imag
        
    return rffarr