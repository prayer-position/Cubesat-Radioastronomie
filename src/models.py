import torch 
import torch.nn as nn

class GMSKAutoEncoder(nn.Module):
    def __init__(self):
        super(GMSKAutoEncoder, self).__init__()

        # Encoder : Compresses noisy I & Q stream into features
        self.encoder = nn.Sequential(
            # Input shape: (batch_size, 2, signal_length)
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Decoder : Reconstructs clean I & Q stream from features
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Conv1d(8, 2, kernel_size=3, padding=1), # Output shape: (batch_size, 2, signal_length)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def summary(self, input_size = (2, 2048)):
        from torchsummary import summary
        summary(self, input_size=input_size)  



class GMSK_VAE(nn.Module):
    def __init__(self, latent_channels = 16):
        super(GMSK_VAE, self).__init__()

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Final Encoder Layer: outputs 2x the channels (Mu + LogVar)
            nn.Conv1d(32, 2, * latent_channels, kernel_size = 3, stride = 1, padding = 1),
        )

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 2, kernel_size = 3, padding = 1)
        )
    
    def reparameterize(self, mu, logvar):
        """ 
        The Magic Trick:
        z = mu + sigma*epsilon
        """
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we just use the mean (most likely value)
            return mu

    def forward(self, x):
        # Encode
        features = self.encoder(x)

        # Split into mu & logvar (along channel dimension)
        mu, logvar = torch.chunk(features, 2, dim = 1)

        # Sample Z
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar