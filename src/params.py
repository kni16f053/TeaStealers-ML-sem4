from dataclasses import dataclass

# Параметры эксперимента !! менять только поле experiment_name !!
@dataclass(frozen=True)
class Settings:
    mlflow_uri: str = "http://localhost:8090"
    tracking_commit_file: str = "../tracking_commit.lock"
    experiment_name: str = "Model v1"
    run_name: str = "Sample"

# Параметры датасетов !! лучше ничего не менять !!
@dataclass(frozen=True)
class Dataset:
    path: str = "data/external/labeled_dataset.txt"
    delimiter: str = "№"
    audio_dir: str = "data/external/audios/"


@dataclass(frozen=True)
class PreparedDataset:
    save_path: str = "data/prepared/prepared_dataset.csv"
    vocab_path: str = "data/prepared/phonemes_vocab.csv"
    sep: str = "№"
    
@dataclass(frozen=True)
class AugmentedDataset:
    path: str = "data/interim/augmented_dataset.csv"
    audio_dir: str = "data/interim/augmented_audios/"
    sep: str = "№"
    
@dataclass(frozen=True)
class ProcessedDataset:
    save_path: str = "data/processed/full_dataset.txt"
    sep: str = "№"
    
# Параметры обучения !! Можно менять всё) !!
@dataclass(frozen=True)
class Dataloader:
    train_split: float = 0.001
    val_split: float = 0.001
    num_workers: int = 16
    batch_size: int = 1
    
@dataclass(frozen=True)
class Model:
    name: str = "facebook/wav2vec2-large-960h-lv60"
    save_path: str = "models/v1/"
    checkpoint_name: str = "sample"
    hidden_size: int = 1024
    
@dataclass(frozen=True)
class Optimizer:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    
@dataclass(frozen=True)
class Training:
    num_epochs: int = 1
    
@dataclass(frozen=True)
class Evaluation:
    metric: str = "wer"

