import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import librosa
import json
import numpy as np
from torch.nn import Linear
from evaluate import load
import mlflow
from mlflow.models import infer_signature


from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer
    )

from torch.optim import AdamW

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')    

from ..params import PreparedDataset, ProcessedDataset

import sys
sys.path.append('..')

from dataclasses import asdict

from .. import params
from ..data_tools.data_utils import ASR_Dataset, collate_fn
from ..params import Settings, Dataloader, Model, Optimizer, Training, Evaluation

def run_epoch(model, dataloader, processor, train=False, optimizer=None):
    
    total_loss = 0
    metric = 0
    
    if train:
        model.train()
    else:
        model.eval()
        
    for input_values, labels, transcriptions in tqdm(dataloader):
        
        input_values = input_values.to(device)
        labels = labels.to(device)
        
        inputs = {
            "input_values": input_values,
            "labels": labels
        }
        
        if train:
            outputs = model(**inputs)
            loss = outputs.loss
            
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
        
        if train:
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.detach().cpu().item()
        
        predicted_ids = torch.argmax(outputs.logits, dim=-1)
        predicted_transcriptions = processor.batch_decode(predicted_ids)
        
        metric += wer_metric.compute(predictions=predicted_transcriptions, references=transcriptions)
        
    total_loss /= len(dataloader)
    metric /= len(dataloader)
    
    return total_loss, metric

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Model.name)

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=PreparedDataset.vocab_path, 
    unk_token="<unk>", 
    pad_token="<pad>",
    blank_token="<blank>",
    word_delimiter_token=None,
    bos_token=None, 
    eos_token=None,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)  

dataset = ASR_Dataset(transcriptions_df_path=ProcessedDataset.save_path, delimiter=ProcessedDataset.sep, processor=processor)

train_size = int(Dataloader.train_split * len(dataset))
val_size = int(Dataloader.val_split * len(dataset))
test_size = len(dataset) - train_size - val_size

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Wav2Vec2ForCTC.from_pretrained(
    Model.name, 
    vocab_size=len(tokenizer.get_vocab()),
    pad_token_id=tokenizer.pad_token_id,
    ignore_mismatched_sizes=True,
    hidden_size=Model.hidden_size
    )

model.config.bos_token_id = None
model.config.eos_token_id = None
model.config.word_delimiter_token = None

wer_metric = load(Evaluation.metric)

optimizer = AdamW(
    model.parameters(), 
    lr=Optimizer.lr,
    betas=(Optimizer.beta1, Optimizer.beta2),
    weight_decay=Optimizer.weight_decay
    )

if __name__ == "__main__":
    
    mlflow.set_tracking_uri(Settings.mlflow_uri)
    mlflow.set_experiment(Settings.experiment_name)
    
    with mlflow.start_run() as run:
        with open(Settings.tracking_commit_file, "w") as f:
            f.write(f"{run.info.run_id}")
            
        dataLoader_params = {f"Dataloader.{k}":v for k, v in asdict(params.Dataloader()).items()}
        model_params = {f"Model.{k}":v for k, v in asdict(params.Model()).items()}
        optimizer_params = {f"Optimizer.{k}": v for k, v in asdict(params.Optimizer()).items()}
        training_params = {f"Training.{k}": v for k, v in asdict(params.Training()).items()}
        evaluation_params = {f"Evaluation.{k}": v for k, v in asdict(params.Evaluation()).items()}
        mlflow.log_params(dataLoader_params | model_params | optimizer_params | training_params | evaluation_params)

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=Dataloader.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=Dataloader.num_workers
            )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=Dataloader.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=Dataloader.num_workers
            )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=
            Dataloader.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=Dataloader.num_workers
            )

        for name, param in model.named_parameters():
            param.requires_grad = any(unfreeze_name in name for unfreeze_name in ["encoder.layers.23", "lm_head"])
            if param.requires_grad:
                print(f"Разморожен: {name}")
            else:
                print(f"Заморожен: {name}")

        model.to(device)
        
        for epoch in range(Training.num_epochs):

            train_loss, train_wer = run_epoch(model, dataloader=train_dataloader, processor=processor, train=True, optimizer=optimizer)
            mlflow.log_metric("Train_loss", train_loss, step=epoch)
            mlflow.log_metric("Train_wer", train_wer, step=epoch)
            
            print(f"Epoch {epoch + 1} train_loss = {train_loss}, train_wer = {train_wer}")
            
            val_loss, val_wer = run_epoch(model, dataloader=val_dataloader, processor=processor, train=False, optimizer=None)
            mlflow.log_metric("Validation_loss", val_loss, step=epoch)
            mlflow.log_metric("Validation_wer", val_wer, step=epoch)
            
            print(f"Epoch {epoch + 1} validation_loss = {val_loss}, validation_wer = {val_wer}")
        
        torch.save(model.state_dict(), Model.save_path + f"{Model.checkpoint_name}.pth")
        

        test_loss, test_wer = run_epoch(model, dataloader=test_dataloader, processor=processor, train=False, optimizer=None)

        print(f"test_loss = {test_loss}, test_wer = {test_wer}")
        
        mlflow.log_metric("Test_loss", test_loss, step=epoch)
        mlflow.log_metric("Test_wer", test_wer, step=epoch)
        
        
