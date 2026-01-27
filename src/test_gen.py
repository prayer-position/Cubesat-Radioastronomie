import numpy as np
from src.modulation import gmsk_mod
from src.channel import awgn

def generate_test_signal(snr, L, fc):
    """
    Generates a single test signal burst for evaluation.
    
    Parameters:
        snr: SNR value for the noisy signal.
        L, fc: GMSK parameters.
    Returns:
        (bits, s_noisy, s_clean): Tuple of noisy and clean signal arrays.
    """
    bits_per_sample = 256
    # 1. Generate random bits
    a = np.random.randint(2, size=bits_per_sample)
    
    # 2. Modulate (Clean Signal)
    _, s_clean = gmsk_mod(a, fc, L, BT=0.3)
    
    # 3. Add Noise (Noisy Signal)
    s_noisy = awgn(s_clean, snr, L=L)
    
    return a, s_noisy, s_clean