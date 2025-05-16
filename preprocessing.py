## preprocessing.py

from typing import Union
import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a 1D ECG signal.

    Args:
        signal (np.ndarray): 1D numpy array of ECG samples.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Filter order. Default is 4.

    Returns:
        np.ndarray: Filtered signal of the same shape and dtype as input.

    Raises:
        ValueError: If input signal is empty or fs <= 0.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"Input signal must be 1D array, but got {signal.ndim}D."
        )
    if signal.size == 0:
        raise ValueError("Input signal is empty.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if not 0 < low < 1:
        raise ValueError(
            f"Lowcut frequency must be between 0 and Nyquist frequency. Got lowcut={lowcut} Hz with fs={fs} Hz."
        )
    if not 0 < high < 1:
        raise ValueError(
            f"Highcut frequency must be between 0 and Nyquist frequency. Got highcut={highcut} Hz with fs={fs} Hz."
        )
    if low >= high:
        raise ValueError(
            f"Lowcut frequency must be less than highcut frequency. Got lowcut={lowcut} Hz, highcut={highcut} Hz."
        )

    b, a = butter(order, [low, high], btype='bandpass')
    filtered_signal = filtfilt(b, a, signal)

    # Preserve dtype and shape of input
    filtered_signal = filtered_signal.astype(signal.dtype, copy=False)

    return filtered_signal


def normalize(segment: np.ndarray) -> np.ndarray:
    """
    Perform per-segment z-score normalization.

    The output segment has zero mean and unit standard deviation.
    If the input segment has zero standard deviation (constant signal),
    a zero array of the same shape is returned to avoid division by zero.

    Args:
        segment (np.ndarray): 1D numpy array of ECG segment data.

    Returns:
        np.ndarray: Normalized segment, same shape and dtype as input.

    Raises:
        ValueError: If input segment is empty or not 1D.
    """
    if segment.ndim != 1:
        raise ValueError(
            f"Input segment must be 1D array, but got {segment.ndim}D."
        )
    if segment.size == 0:
        raise ValueError("Input segment is empty.")

    mean_val = np.mean(segment)
    std_val = np.std(segment)

    if std_val < 1e-10:
        # Handle zero or near-zero std deviation (constant signal)
        normalized = np.zeros_like(segment)
    else:
        normalized = (segment - mean_val) / std_val

    # Preserve dtype of input segment
    normalized = normalized.astype(segment.dtype, copy=False)

    return normalized
