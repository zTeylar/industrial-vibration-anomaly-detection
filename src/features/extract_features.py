from pathlib import Path
import numpy as np
import pandas as pd
import torch
import h5py
import json
import argparse
import re
import datetime
import logging
import hashlib
from src.config.config_loader import load_config


logger = logging.getLogger(__name__)


def extract_datetime(pathname):

    pattern = r"\b\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\b"
    match = re.search(pattern, pathname)
    if match:
        dt_format = "%Y.%m.%d.%H.%M.%S"
        return datetime.datetime.strptime(match.group(), dt_format)
    else:
        raise ValueError(f"Incorrect datetime format in pathname: {pathname}, correct format should be YYYY.MM.DD.HH.MM.SS")


def compute_stft(signal, window_size, hop_length, window):
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    stft_transform = torch.stft(signal_tensor,
                                n_fft=window_size,
                                hop_length=hop_length,
                                window=window,
                                return_complex=True)

    stft_magnitude = torch.abs(stft_transform)
    stft_power = stft_magnitude ** 2
    stft_spec_db = 10.0 * torch.log10(stft_power + 1e-10)

    return stft_spec_db.numpy()


def validate_file(data, config, filename):
    if data.shape != (config.dataset.num_channels, config.dataset.n_samples):
        logger.warning(f"[SKIP] Bad shape {data.shape} in file: {filename}")
        return False
    if np.isnan(data).any() or np.isinf(data).any():
        logger.warning(f"[SKIP] Corrupted math (NaN/Inf) in file: {filename}")
        return False

    return True


def process_file(file_path, config, window):
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    data = df.values.T
    if not validate_file(data, config, file_path.name):
        return None
    
    tensor_data = torch.tensor(data, dtype=torch.float32)
    stft_spec_db = compute_stft(tensor_data, config.stft.window_size, config.stft.hop_length, window)

    return stft_spec_db


def compute_hash(file_path, chunk_size=4096):
    hash_sha256 = hashlib.sha256()
    
    if Path(file_path).is_dir():
        for file_path in sorted(Path(file_path).iterdir()):
            file_hash = compute_hash(file_path, chunk_size)
            hash_sha256.update(file_hash.encode())
    else:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def create_hdf5_dataset(config):
    files = sorted([f for f in config.dataset.data_dir.iterdir() if f.is_file() and not f.name.startswith('.')],
                    key=lambda x: extract_datetime(x.name))
    window = torch.hann_window(config.stft.window_size)
    dataset_hash = compute_hash(config.dataset.data_dir)

    with h5py.File(config.hdf5_file, 'w') as f:
        dataset = None
        manifest = {
            "dataset_hash": dataset_hash,
            "files": {}
        }
        
        for idx, file in enumerate(files):
            spec = process_file(file, config, window)
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
            manifest[file.name] = {
                "dataset_index": dataset.shape[0] - 1, 
                "file_hash": compute_hash(file)
                }
        if dataset is None:
            logger.warning("No valid files were processed. HDF5 dataset was not created.")
            return
        
        logger.info(f"Created dataset with shape: {dataset.shape}")

    with open(config.manifest_file, 'w') as mf:
        json.dump(manifest, mf, indent=4)


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("feature_extraction.log"),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Extract features and create HDF5 dataset")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process from YAML file (e.g., set_1)')
    args = parser.parse_args()
    config = load_config(dataset_name=args.dataset)
    create_hdf5_dataset(config)
