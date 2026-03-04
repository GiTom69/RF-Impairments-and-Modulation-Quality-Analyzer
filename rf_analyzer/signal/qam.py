import numpy as np

def bits_to_groups(bits: np.ndarray, group_size: int) -> np.ndarray:
    # Ensure the number of bits is divisible by the group size, if not, pad with zeros
    if len(bits) % group_size != 0:
        padding_size = group_size - (len(bits) % group_size)
        bits = np.pad(bits, (0, padding_size), mode='constant')
    
    return bits.reshape(-1, group_size)

def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    groups = bits_to_groups(bits, 4)
    level_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1): 1,
        (1, 0): 3,
    }

    i_levels = np.array([level_map[(int(g[0]), int(g[1]))] for g in groups], dtype=float)
    q_levels = np.array([level_map[(int(g[2]), int(g[3]))] for g in groups], dtype=float)

    symbols = (i_levels + 1j * q_levels) / np.sqrt(10.0)
    return symbols