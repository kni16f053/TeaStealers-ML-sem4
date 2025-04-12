import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np

class ASR_Dataset(Dataset):
    
    def __init__(self, transcriptions_df_path, delimiter, processor, max_len=16000*10):
        super().__init__()
        
        transcriptions_df = pd.read_csv(transcriptions_df_path, delimiter=delimiter)
                
        self.audio_paths = list(transcriptions_df["path"])
        self.transcriptions = list(transcriptions_df["transcription"])
        
        self.processor = processor
        
        self.max_len = max_len
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        
        audio_path, transcription = self.audio_paths[index], self.transcriptions[index]
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        if len(audio.shape) > 2:
            audio = np.mean(audio, axis=1)
        
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.processor(text=transcription).input_ids
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.int),
            "transcription": transcription
        }