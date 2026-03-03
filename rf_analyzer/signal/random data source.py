import numpy as np
import random

def generate_random_signal(length: int, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.random.uniform(-1, 1, size=length)

def generate_random_bytes(length: int) -> np.ndarray:
    return np.array([random.getrandbits(8) for _ in range(length)], dtype=np.uint8)