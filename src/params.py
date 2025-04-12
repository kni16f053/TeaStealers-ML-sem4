from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_uri: str = "http://localhost:8090"
    tracking_commit_file: str = "../tracking_commit.lock"


@dataclass(frozen=True)
class Dataset:
    path: str = "data/external/labeled_dataset.txt"
    delimiter: str = " № "


@dataclass(frozen=True)
class PreparedDataset:
    save_path: str = "data/prepared/prepared_dataset.csv"
    vocab_path: str = "data/prepared/phonemes_vocab.csv"

