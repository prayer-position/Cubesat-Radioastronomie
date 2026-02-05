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