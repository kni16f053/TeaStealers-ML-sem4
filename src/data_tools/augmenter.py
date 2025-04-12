# Скрипт для аугментации аудиофайлов
# __________________________________
#
# На первом шаге считывается csv-файл со словом, его транскрипцией, полом говорящего и ссыдкой на аудио
#
# Далее каждому файлу присваивается соответствующий путь к файлу, лежащему в папке ./audios
#
# Затем к каждой из аудиозаписей применяются аугментации: 
# добавление шума, малые изменение тона, ускорение/замедление, увеличение/уменьшение громкости
# 
# При этом для женских и мужских количество добавляемых файлов на одну аудиозапись разное и составляет
# 11 и 3, соответственно (этот параметр настраивается) - это нужно для балансировки числа мужских и женских 
# записей
#
# Все аугментированные файлы сохраняются в папке ./augmented_audios и добавляются соответствующие строки в
# csv-файл с аугментированным датасетом (augmented_dataset.csv)
# __________________________________

import soundfile as sf
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, Gain, TimeStretch, PitchShift
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


dataset_file = "prepared_dataset.csv"
augmented_dataset_file = "augmented_dataset.csv"


male_aug = 3
female_aug = 11
aug_dir = "augmented_audios/"


def make_path(url):
    return "audios/" + url.split("/")[-1]


augmenter = Compose([
    AddGaussianNoise(p=0.5), # Небольшой шум
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5), # Малое изменение тона
    TimeStretch(min_rate=0.5, max_rate=2, p=0.5, leave_length_unchanged=False), # небольшое ускорение / замедление
    Gain(min_gain_db=-12, max_gain_db=20, p=0.5), # небольшое изменение грмокости
])


if __name__ =="__main__":
    
    df = pd.read_csv(dataset_file, delimiter="№")
    
    df["path"] = df["url"].apply(make_path)

    columns = {
    "word": [],
    "transcription": [],
    "gender": [],
    "url": [],
    "path": []
}
    
    df_augmented = pd.DataFrame(columns)

    for i in tqdm(range(len(df))):
        
        row = df.iloc[[i]]
        
        audio, sr = sf.read(row["path"].iloc[0])
        
        if len(audio.shape) > 1: # Если аудио многоканальное, то усредняем его по каналам
            audio = audio.mean(axis=1)
        
        n_aug = male_aug if row["gender"].iloc[0] == "мужчина" else female_aug
        
        for j in range(n_aug):
            
            file_name = f"aug_{j}_" + row["path"].iloc[0].split("/")[-1]
            
            path = aug_dir + file_name
            
            augmented_audio = augmenter(samples=audio, sample_rate=sr)
            sf.write(path, augmented_audio, sr)
            
            new_row = row.copy()
            new_row["path"] = path
            
            df_augmented = pd.concat([df_augmented, new_row])
    
    del df_augmented["Unnamed: 0"]
    del df_augmented["Unnamed: 0.1"]
    df_augmented.to_csv(augmented_dataset_file, sep="№")
