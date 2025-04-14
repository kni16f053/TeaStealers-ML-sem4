import pandas as pd
import torch
import librosa
import numpy as np

import sys
sys.path.append('..')



from ..params import Settings, Model, PreparedDataset, ProcessedDataset

import mlflow

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(Settings.mlflow_uri)
client = MlflowClient()
experiment_name = "Model v1"

experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id]
)


from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    )

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Model.name)

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=PreparedDataset.vocab_path, 
    unk_token="<unk>", 
    pad_token="<pad>",
    word_delimiter_token=None,
    bos_token=None, 
    eos_token=None,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Wav2Vec2ForCTC.from_pretrained(
    Model.name, 
    vocab_size=len(tokenizer.get_vocab()),
    pad_token_id=tokenizer.pad_token_id,
    ignore_mismatched_sizes=True)

model.config.bos_token_id = None
model.config.eos_token_id = None
model.config.word_delimiter_token = None

model.state_dict = torch.load(Model.save_path + Model.checkpoint_name + ".pth")
model.to(device);

model.eval()

df = pd.read_csv(ProcessedDataset.save_path, delimiter=ProcessedDataset.sep, engine="python")

def show_result(model, df, word, processor):
    
    df_word = df[df["word"] == word]
    
    transcription = list(df_word["transcription"])[0]
    audio_path = list(df_word["path"])[0]
    
    
    audio, sr = librosa.load(audio_path, sr=16000)
        
    if len(audio.shape) > 2:
        audio = np.mean(audio, axis=1)
    
    audio_inputs = processor(
        audio=audio,
        sampling_rate=sr,
        max_length=16000*10,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_values.to(device)
    
    labels = processor(text=transcription).input_ids
        
    inputs = {
    "input_values": audio_inputs,
    "labels": torch.tensor(labels, dtype=torch.int)
    }
    
    outputs = model(**inputs).logits
    predicted_ids = torch.argmax(outputs, dim=-1)
    predicted_transcription = processor.batch_decode(predicted_ids)[0]
    
    print(f"Word: \t\t   {word}")
    print(f"Truth:\t\t / {transcription} /")
    print(f"Predicted:\t / {predicted_transcription} /")
    print("\n")
    
if __name__ == "__main__":
    
    original_stdout = sys.stdout

    with open(Model.save_path + Model.checkpoint_name + f"_example.txt", "w") as f:
        sys.stdout = f
          
        show_result(model, df, "cover", processor)
        show_result(model, df, "damage", processor)
        show_result(model, df, "awful", processor)
        show_result(model, df, "zoo", processor)
        show_result(model, df, "further", processor)
        show_result(model, df, "baby", processor)
        
    sys.stdout = original_stdout
    
    show_result(model, df, "cover", processor)
    show_result(model, df, "damage", processor)
    show_result(model, df, "awful", processor)
    show_result(model, df, "zoo", processor)
    show_result(model, df, "further", processor)
    show_result(model, df, "baby", processor)
    
    run_id = runs[0].info.run_id
    
    client.log_artifact(run_id, Model.save_path + Model.checkpoint_name + f"_example.txt", "")