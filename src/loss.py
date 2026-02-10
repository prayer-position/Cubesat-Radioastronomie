import torch
import torch.nn as nn

def vae_loss_function(recon_x, x, mu, logvar, beta = 0.001):
    """
    beta: Weight of the KL term.
          For denoising, keep beta SMALL 
          If beta is too high, the model ignores the input and outputs pure noise
    """
    # Reconstruction Loss (MSE)
    MSE = nn.functional.mse_loss(recon_x, x, reduction = 'sum')

    # KL Divergence
    # Measures how much the learned distribution diverges from N(0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + (beta * KLD), MSE, KLD