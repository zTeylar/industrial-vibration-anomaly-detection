from pathlib import Path
import numpy as np
import pandas as pd
import torch
import h5py
import json


DATA_DIR = Path('data/raw/1st_test')
SAMPLE_RATE = 20480
WINDOW_SIZE = 1024
HOP_LENGTH = 512
HDF5_FILE = 'bearing_data.h5'


stft_window = torch.hann_window(WINDOW_SIZE)


def process_file(file_path):
    """Process file and return a list of mel spectrograms in dB."""
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    data = df.values.T

    if data.shape != (8, SAMPLE_RATE):
        print(f"\n[SKIP] Bad shape {data.shape} in file: {file_path.name}")
        return None
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"\n[SKIP] Corrupted math (NaN/Inf) in file: {file_path.name}")
        return None
    
    tensor_data = torch.tensor(data, dtype=torch.float32)

    stft_transform = torch.stft(tensor_data,
                                n_fft=WINDOW_SIZE,
                                hop_length=HOP_LENGTH,
                                window=stft_window,
                                return_complex=True)

    stft_magnitude = torch.abs(stft_transform)
    stft_power = stft_magnitude ** 2
    stft_spec_db = 10.0 * torch.log10(stft_power + 1e-10)

    return stft_spec_db.numpy()


def create_hdf5_dataset():
    """Create an HDF5 dataset containing mel spectrograms from files."""
    files = sorted([f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith('.')])

    with h5py.File(HDF5_FILE, 'w') as f:
        dataset = None
        manifest = {}

        for idx, file in enumerate(files):
            spec = process_file(file)
            if spec is None:
                continue
            if dataset is None:
                c, h, w = spec.shape
                dataset = f.create_dataset(
                    'stft_spectrograms',
                    shape=(0, c, h, w),
                    maxshape=(None, c, h, w),
                    dtype=np.float32,
                    chunks=(1, c, h, w)
                )

            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = spec
            manifest[file.name] = dataset.shape[0] - 1
        print(f"Created dataset with shape: {dataset.shape}")

        with open('manifest.json', 'w') as mf:
            json.dump(manifest, mf, indent=4)

create_hdf5_dataset()
