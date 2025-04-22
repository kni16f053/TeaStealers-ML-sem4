import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import itertools

class ASR_Dataset(Dataset):
    
    def __init__(self, transcriptions_df_path, delimiter, processor, max_len=16000*10):
        super().__init__()
        
        transcriptions_df = pd.read_csv(transcriptions_df_path, delimiter=delimiter)
                
        self.audio_paths = list(transcriptions_df["path"])
        self.transcriptions = list(transcriptions_df["transcription"])
        
        self.processor = processor
        
        self.max_len = max_len
        
        self.cache = {}
        self.is_cached = False
        
    def __len__(self):
        return len(self.audio_paths)
    
    def set_cache_flag(self, flag):
        self.isCached = flag

    def get_cache_flag(self):
        return self.is_cached
    
    def __getitem__(self, index):
        
        transcription = self.transcriptions[index]
                
        if not(self.is_cached):
            
            audio_path = self.audio_paths[index]
            audio, sr = librosa.load(audio_path, sr=16000)       
            if len(audio.shape) > 2:
                audio = np.mean(audio, axis=1)
            self.cache[index] = {"audio": audio, "sr": sr}
            
        else:
            audio, sr = self.cache[index]["audio"], self.cache[index]["sr"]
        
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            max_length=self.max_len,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.processor(text=transcription).input_ids
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.int),
            "transcription": transcription
        }
        
def collate_fn(batch): # при batch_size > 1 не будет работать!!!
    
    # input_values = torch.stack([element["input_values"] for element in batch])
    # labels = [element["labels"] for element in batch]
    # transcriptions = [element["transcription"] for element in batch]
    
    # labels = pad_sequence(
    #     labels,
    #     batch_first=True,
    #     padding_value=2
    # )
    
    element = batch[0]
    input_values = element["input_values"].unsqueeze(0)
    labels = element["labels"].unsqueeze(0)
    transcriptions = [element["transcription"]]
    
    return input_values, labels, transcriptions

def ctc_decode(pred_ids, processor):
    pred_ids = [
                    p for p, _ in itertools.groupby(pred_ids) 
                    if p not in [processor.tokenizer.pad_token_id, processor.tokenizer.unk_token_id]
               ]
    return processor.decode(pred_ids)