import numpy as np
import os
from src.modulation import gmsk_mod
from src.channel import awgn

def generate_denoising_dataset(N_samples=1000, bits_per_sample=256, L=16, fc=146e6, BT=0.3):
    """
    Generates training pairs (Noisy_IQ, Clean_IQ) for an Auto-Encoder.
    
    Parameters:
        N_samples: Number of signal bursts to generate.
        bits_per_sample: Number of bits in each burst.
        L, fc, BT: GMSK parameters.
    """
    X_noisy = []
    Y_clean = []
    
    # Range of SNRs to make the AI robust (Low to High)
    snr_range = np.linspace(-5, 15, N_samples) 
    
    print(f"Generating {N_samples} signal bursts...")

    for i in range(N_samples):
        # Generate random bits
        a = np.random.randint(2, size=bits_per_sample)
        
        # Modulate (Clean Signal)
        _, s_complex = gmsk_mod(a, fc, L, BT)
        
        # Add Noise (Noisy Signal)
        s_noisy = awgn(s_complex, snr_range[i], L=L)

        max_val = np.max(np.abs(s_noisy))
        if max_val > 0:
            s_noisy = s_noisy / max_val
        
        # 4. Reformat to (2, Time) for Neural Network (Real/Imag channels)
        # We split complex into two real channels: [Real, Imag]
        clean_input = np.vstack((np.real(s_complex), np.imag(s_complex)))
        noisy_input = np.vstack((np.real(s_noisy), np.imag(s_noisy)))
        
        X_noisy.append(noisy_input)
        Y_clean.append(clean_input)

    # Convert to numpy arrays for storage
    X_noisy = np.array(X_noisy, dtype=np.float32)
    Y_clean = np.array(Y_clean, dtype=np.float32)

    # Create directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save as compressed numpy files
    np.save('data/X_noisy.npy', X_noisy)
    np.save('data/Y_clean.npy', Y_clean)
    
    print(f"Dataset saved! Shapes: X={X_noisy.shape}, Y={Y_clean.shape}")

if __name__ == "__main__":
    generate_denoising_dataset()

