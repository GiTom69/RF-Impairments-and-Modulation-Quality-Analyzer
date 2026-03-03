# inputs other than the signal:
# - sampling rate
# - number of bits
# - jitter (in a future version)

# all others are derived from these two parameters:
# - quantization step size
# - ENOB

import numpy as np

def quantize(sample:float, SR:float, n_bits:int) -> float:
    clipped_signal = np.clip(sample, -1, 1)

    # quantization step size
    q_step = 2**(1 - calculate_enob(SR, n_bits)) # assuming the signal is normalized to [-1, 1]
    
    # quantize the signal
    quantized_signal = np.round(clipped_signal / q_step) * q_step
    
    return quantized_signal

def calculate_enob(SR:float, n_bits:int) -> float:
    return n_bits - np.log2(SR / (SR / 2))

def calculate_sqnr(SR:float, n_bits:int) -> float:
    return 6.02 * calculate_enob(SR, n_bits) + 1.76 # in dB