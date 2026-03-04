import numpy as np

def compute_fft(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of a signal and return the frequency bins and their corresponding magnitudes.

    Parameters:
    signal (np.ndarray): The input signal to analyze.
    sampling_rate (float): The sampling rate of the signal in Hz.

    Returns:
    freqs (np.ndarray): The frequency bins corresponding to the FFT output.
    magnitudes (np.ndarray): The magnitudes of the FFT output.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    
    # Get the frequency bins
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Get the magnitudes of the FFT
    magnitudes = np.abs(fft_result) / n  # Normalize by the number of samples
    
    return freqs, magnitudes

def compute_power_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of a signal.

    Parameters:
    signal (np.ndarray): The input signal to analyze.
    sampling_rate (float): The sampling rate of the signal in Hz.

    Returns:
    freqs (np.ndarray): The frequency bins corresponding to the power spectrum.
    power_spectrum (np.ndarray): The power spectrum of the signal.
    """
    freqs, magnitudes = compute_fft(signal, sampling_rate)
    
    # Power spectrum is the square of the magnitudes
    power_spectrum = magnitudes ** 2
    
    return freqs, power_spectrum

def compute_spectrogram(signal: np.ndarray, sampling_rate: float, window_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spectrogram of a signal.

    Parameters:
    signal (np.ndarray): The input signal to analyze.
    sampling_rate (float): The sampling rate of the signal in Hz.
    window_size (int): The size of the window for FFT in samples.
    hop_size (int): The hop size between windows in samples.

    Returns:
    freqs (np.ndarray): The frequency bins corresponding to the spectrogram.
    times (np.ndarray): The time bins corresponding to the spectrogram.
    spectrogram (np.ndarray): The spectrogram of the signal.
    """
    n = len(signal)
    num_windows = (n - window_size) // hop_size + 1
    
    spectrogram = []
    
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        windowed_signal = signal[start:end] * np.hanning(window_size)  # Apply Hanning window
        freqs, magnitudes = compute_fft(windowed_signal, sampling_rate)
        spectrogram.append(magnitudes ** 2)  # Power spectrum
    
    spectrogram = np.array(spectrogram).T  # Transpose to get frequencies as rows and time as columns
    times = np.arange(num_windows) * (hop_size / sampling_rate)
    
    return freqs, times, spectrogram

def compute_phase_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the phase spectrum of a signal.

    Parameters:
    signal (np.ndarray): The input signal to analyze.
    sampling_rate (float): The sampling rate of the signal in Hz.

    Returns:
    freqs (np.ndarray): The frequency bins corresponding to the phase spectrum.
    phase_spectrum (np.ndarray): The phase spectrum of the signal in radians.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    
    # Get the frequency bins
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Get the phase spectrum
    phase_spectrum = np.angle(fft_result)
    
    return freqs, phase_spectrum

def compute_OBW(signal: np.ndarray, sampling_rate: float, power_threshold: float = 0.99) -> float:
    """
    Compute the Occupied Bandwidth (OBW) of a signal.

    Parameters:
    signal (np.ndarray): The input signal to analyze.
    sampling_rate (float): The sampling rate of the signal in Hz.
    power_threshold (float): The percentage of total power to consider for OBW calculation (default is 0.99 for 99%).

    Returns:
    obw (float): The occupied bandwidth in Hz.
    """
    freqs, power_spectrum = compute_power_spectrum(signal, sampling_rate)
    
    total_power = np.sum(power_spectrum)
    cumulative_power = np.cumsum(power_spectrum)
    
    # Find the frequency bins that contain the specified percentage of total power
    lower_idx = np.searchsorted(cumulative_power, (1 - power_threshold) * total_power)
    upper_idx = np.searchsorted(cumulative_power, power_threshold * total_power)
    
    obw = freqs[upper_idx] - freqs[lower_idx]
    
    return obw

def compute_SNR(signal: np.ndarray) -> float:
    # compute the OBW of the signal
    obw = compute_OBW(signal, sampling_rate=1.0)  # Assuming normalized frequency for OBW calculation
    