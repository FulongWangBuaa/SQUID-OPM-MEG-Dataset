import numpy as np
import mne

def amplitude_spectrum(X, sfreq=1000):
    sfreq = sfreq
    if X.ndim == 1:
        n = len(X)
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量  
        Y = np.fft.fft(X)  # 计算FFT  
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[:n//2]
    else:
        n = X.shape[-1]
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量
        Y = np.fft.fft(X,axis=-1)
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[...,:n//2]

