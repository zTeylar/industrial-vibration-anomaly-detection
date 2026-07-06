import json
import torch
from torch import nn
import h5py


class WeightedMSELoss(nn.Module):
    def __init__(self, penalty_factor=1.0):
        super(WeightedMSELoss, self).__init__()
        self.penalty_factor = penalty_factor
        

    def forward(self, input, target):
        mse = torch.square(input - target)
        weight = 1 + (self.penalty_factor * target)
        weighted_mse = weight * mse
        return torch.mean(weighted_mse)
    

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, num_channels):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels=num_channels)
        self.decoder = Decoder(out_channels=num_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class h5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, h5_metadata):
        self.h5_file = h5_file
        self.h5_metadata = h5_metadata
        with h5py.File(self.h5_file, 'r') as f:
            self.dataset_length = len(f['stft_spectrograms'])
        with open(self.h5_metadata, 'r') as f:
            h5_metadata = json.load(f)
            self.min_db = h5_metadata['min_db']
            self.max_db = h5_metadata['max_db']
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            sample = f['stft_spectrograms'][idx]
        torch_sample = torch.tensor(sample, dtype=torch.float32)
        torch_sample_scaled = (torch_sample - self.min_db) / (self.max_db - self.min_db)
        return torch_sample_scaled, torch_sample_scaled


def test_encoder_decoder():
    model = Encoder()
    input_tensor = torch.randn(1, 8, 513, 41)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    model2 = Decoder()
    output_tensor2 = model2(output_tensor)
    print(output_tensor2.shape)
    