import numpy as np

def bits_to_groups(bits: np.ndarray, group_size: int) -> np.ndarray:
    # Ensure the number of bits is divisible by the group size, if not, pad with zeros
    if len(bits) % group_size != 0:
        padding_size = group_size - (len(bits) % group_size)
        bits = np.pad(bits, (0, padding_size), mode='constant')
    
    return bits.reshape(-1, group_size)

def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    # Group bits into 4
    groups = bits_to_groups(bits, 4)
    
    # Map bits to QAM16 symbol
    symbols = []
    for group in groups:
        I = group[0:2]
        Q = group[2:4]
        
        symbols.append(complex(I, Q))
    
    return np.array(symbols)