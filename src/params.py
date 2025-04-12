from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_uri: str = "http://localhost:8090"
    tracking_commit_file: str = "../tracking_commit.lock"
    experiment_name: str = "Model v1"


@dataclass(frozen=True)
class Dataset:
    path: str = "data/external/labeled_dataset.txt"
    delimiter: str = " № "
    audio_dir: str = "data/external/audios/"


@dataclass(frozen=True)
class PreparedDataset:
    save_path: str = "data/prepared/prepared_dataset.csv"
    vocab_path: str = "data/prepared/phonemes_vocab.csv"
    sep: str = "№"
    
@dataclass(frozen=True)
class Split:
    train: float = 0.001
    val: float = 0.001
    
@dataclass(frozen=True)
class Model:
    name: str = "facebook/wav2vec2-large-960h-lv60"
    
@dataclass(frozen=True)
class Optimizer:
    lr: float = 1e-3
    
@dataclass(frozen=True)
class Training:
    batch_size: int = 2
    num_epochs: int = 2
    
@dataclass(frozen=True)
class Evaluation:
    metric: str = "wer"

