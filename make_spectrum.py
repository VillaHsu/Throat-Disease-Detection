import numpy as np
import scipy
import scipy.io.wavfile as wav
from scipy import signal
#from pylab import specgram
#import matplotlib.pyplot as plt
import os
#import cPickle as pickle
import h5py
import librosa

FRAMESIZE = 1024
OVERLAP = 80
FFTSIZE = 400

wav,sr=librosa.load("./St.wav",sr=1600000)
stft =librosa.stft(wav,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE)
Sxx =np.log10(np.abs(stft)) # Log-Power Spectrum

print(Sxx)
