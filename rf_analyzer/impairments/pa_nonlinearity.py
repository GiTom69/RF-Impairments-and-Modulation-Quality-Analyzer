import numpy as np

def PA_amplify(signal: np.ndarray, gain_lin: float, threshold: float = 1.0) -> np.ndarray:   
    return np.clip(gain_lin * signal, -threshold, threshold)

def PA_amplify_custom( signal: np.ndarray, gain_lin: float, threshold: float = 1.0, amp_curve: callable = None) -> np.ndarray:
    # the amp curve function should map the input range [-1, 1] to the output range [-1, 1]
    if amp_curve is None:
        raise ValueError("amp_curve function must be provided for custom PA amplification.")
    
    return np.clip(gain_lin * amp_curve(signal), -threshold, threshold)