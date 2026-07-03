from pathlib import Path
import yaml
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src/config/pipeline_config.yaml"

@dataclass
class DatasetConfig:
    data_dir: Path
    num_channels: int
    n_samples: int

@dataclass
class STFTConfig:
    sample_rate: int
    window_size: int
    hop_length: int

@dataclass
class PipelineConfig:
    dataset: DatasetConfig
    stft: STFTConfig
    hdf5_file: Path
    manifest_file: Path
    metadata_file: Path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH, dataset_name: str = None) -> PipelineConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    dataset_dict = config_dict['datasets'][dataset_name]

    dataset_config = DatasetConfig(
        data_dir=PROJECT_ROOT / dataset_dict['data_dir'],
        num_channels=dataset_dict['num_channels'],
        n_samples=dataset_dict['n_samples']
    )
    stft_config = STFTConfig(**config_dict['stft_params'])
    processed_dir = PROJECT_ROOT / config_dict['storage']['processed_dir_path']
    processed_dir.mkdir(parents=True, exist_ok=True)
    hdf5_file = processed_dir / f"{dataset_name}_data.h5"
    manifest_file = processed_dir / f"{dataset_name}_manifest.json"
    metadata_file = processed_dir / f"{dataset_name}_metadata.json"

    return PipelineConfig(dataset=dataset_config, stft=stft_config, hdf5_file=hdf5_file, manifest_file=manifest_file, metadata_file=metadata_file)
