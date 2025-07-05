import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0 or high >= 1 or low >= high:
        raise ValueError("Invalid cutoff frequencies")
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize(segment):
    if segment.size == 0:
        return segment
        
    mean = np.mean(segment)
    std = np.std(segment)
    
    if std < 1e-10:
        return np.zeros_like(segment)
    
    return (segment - mean) / std