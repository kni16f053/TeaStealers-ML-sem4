import pandas as pd
import pulp
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from ..params import PreparedDataset, AugmentedDataset, ProcessedDataset

df_original = pd.read_csv(PreparedDataset.save_path, delimiter=PreparedDataset.sep, engine="python")

df_original_clean = df_original.copy()
df_original_clean['transcription'] = df_original_clean['transcription'].str.replace(r'[ˈˌ]', '', regex=True)

df_augmented_clean = pd.read_csv(AugmentedDataset.path, delimiter=AugmentedDataset.sep, engine="python")

# df_augmented_clean = df_augmented_clean.drop(['id'], axis=1)
df_augmented_clean = df_augmented_clean[(df_augmented_clean['gender'] == "мужчина") | (df_augmented_clean['gender'] == "женщина")]

def remake_path(path):
    return AugmentedDataset.audio_dir + path.split("/")[-1]

df_augmented_clean["path"] = df_augmented_clean["path"].apply(remake_path)

unary_phonemes_list = [
    "æ","ɪ", "ɛ", "ɒ", "ʊ", "ʌ", "ə",
    "p", "f", "t", "θ", "s", "ʃ", "k", "b", "v", "d", "ð", "z", "ʒ", "ɡ", "h", "m", "n", "ŋ", "r", "l", "w", "j"
]
binary_phonemes_list = [
    "ɑː", "iː", "ɔː", "uː", "ɜː",
    "eɪ", "aɪ", "aʊ", "əʊ", "ɔɪ", "ɪə", "ɛə", "ʊə",
    "tʃ", "dʒ"
]
tokens = unary_phonemes_list + binary_phonemes_list
assert(len(tokens) == 44)

original_cleaned_samples = []
for word, transcription, gender, url, path in zip(df_original_clean['word'],
                                                  df_original_clean['transcription'],
                                                  df_original_clean['gender'],
                                                  df_original_clean['url'],
                                                  df_original_clean['path']):
    transcription_tokens = []
    i = 0
    while i < len(transcription):
        if i < len(transcription) - 1 and transcription[i:i+2] in binary_phonemes_list:
            transcription_tokens += [transcription[i:i+2]]
            i+=2
        else:
            transcription_tokens += [transcription[i]]
            i+=1
    original_cleaned_samples += [{'word': word,
                                'tokens': transcription_tokens,
                                'gender': 'male' if gender == 'мужчина' else 'female',
                                'url': url,
                                'path': path}]

cleaned_samples_to_add = []
for word, transcription, gender, url, path in zip(df_augmented_clean['word'],
                                                  df_augmented_clean['transcription'],
                                                  df_augmented_clean['gender'],
                                                  df_augmented_clean['url'],
                                                  df_augmented_clean['path']):
    transcription_tokens = []
    i = 0
    while i < len(transcription):
        if i < len(transcription) - 1 and transcription[i:i+2] in binary_phonemes_list:
            transcription_tokens += [transcription[i:i+2]]
            i+=2
        else:
            transcription_tokens += [transcription[i]]
            i+=1
    cleaned_samples_to_add += [{'word': word,
                              'tokens': transcription_tokens,
                              'gender': 'male' if gender == 'мужчина' else 'female',
                              'url': url,
                              'path': path}] 
  
male_count = 0
female_count = 0
for sample in cleaned_samples_to_add:
    if sample['gender'] == 'male':
        male_count += 1
    else:
        female_count += 1
        
print(male_count, female_count)

def GetIncludeExclude(tokens, original_samples, samples_to_add):
    N_target = 7500
    token_curr_counts = {} # относится к датасету, к которому мы добавляем
    for token in tokens:
        token_curr_counts[token] = {'total': 0, 'male': 0, 'female': 0}

    for sample in original_samples:
        for token in sample['tokens']:
            token_curr_counts[token]['total'] += 1
            token_curr_counts[token][sample['gender']] += 1
    model = pulp.LpProblem("Dataset_Balancing", pulp.LpMinimize)
    x_add = [pulp.LpVariable(f"x_add_{j}", cat="Binary") for j in range(len(samples_to_add))]
    x_remove = [pulp.LpVariable(f"x_remove_{k}", cat="Binary") for k in range(len(original_samples))]
    d_plus = {t: pulp.LpVariable(f"d_plus_{t}", lowBound=0) for t in tokens}
    d_minus = {t: pulp.LpVariable(f"d_minus_{t}", lowBound=0) for t in tokens}
    u = {t: pulp.LpVariable(f"u_{t}", lowBound=0) for t in tokens}
    v = {t: pulp.LpVariable(f"v_{t}", lowBound=0) for t in tokens}
    alpha = 10  # Вес баланса по количеству
    beta = 1   # Вес баланса по полу
    model += (
        alpha * pulp.lpSum([d_plus[t] + d_minus[t] for t in tokens]) +
        beta * pulp.lpSum([u[t] + v[t] for t in tokens])
    )
    
    for t in tokens:
        sum_add = pulp.lpSum([
            x_add[j] for j, sample in enumerate(samples_to_add)
            if t in sample["tokens"]
        ])
        sum_remove = pulp.lpSum([
            x_remove[k] for k, sample in enumerate(original_samples)
            if t in sample["tokens"]
        ])
        model += (
            token_curr_counts[t]["total"] + sum_add - sum_remove + d_minus[t] - d_plus[t] == N_target
        )
    for t in tokens:
        sum_add_male = pulp.lpSum([
        x_add[j] for j, sample in enumerate(samples_to_add)
        if t in sample["tokens"] and sample["gender"] == "male"
    ])
        sum_add_female = pulp.lpSum([
        x_add[j] for j, sample in enumerate(samples_to_add)
        if t in sample["tokens"] and sample["gender"] == "female"
    ])
        sum_remove_male = pulp.lpSum([
        x_remove[k] for k, sample in enumerate(original_samples)
        if t in sample["tokens"] and sample["gender"] == "male"
    ])
        sum_remove_female = pulp.lpSum([
        x_remove[k] for k, sample in enumerate(original_samples)
        if t in sample["tokens"] and sample["gender"] == "female"
    ])
        model += (
        (token_curr_counts[t]["male"] + sum_add_male - sum_remove_male) -
        (token_curr_counts[t]["female"] + sum_add_female - sum_remove_female) == u[t] - v[t]
    )   
    model.solve()
    print("Status:", pulp.LpStatus[model.status])
    selected_to_add = [j for j in range(len(samples_to_add)) if x_add[j].value() == 1]
    selected_to_remove = [k for k in range(len(original_samples)) if x_remove[k].value() == 1]
    return selected_to_add, selected_to_remove

samples_idx_to_add, samples_idx_to_remove = GetIncludeExclude(tokens, original_cleaned_samples, cleaned_samples_to_add)

print("Добавлено записей:", len(samples_idx_to_add))
print("Удалено записей:", len(samples_idx_to_remove))

samples_idx_to_add = np.array(samples_idx_to_add, dtype=np.int32)
samples_idx_to_remove = np.array(samples_idx_to_remove, dtype=np.int32)

original_samples_np = np.array(original_cleaned_samples, dtype=object)
samples_to_add_np = np.array(cleaned_samples_to_add, dtype=object)

added_samples = samples_to_add_np[samples_idx_to_add]
mask_to_remain = np.ones_like(original_samples_np, dtype=bool)
mask_to_remain[samples_idx_to_remove] = False
remaining_samples = original_samples_np[mask_to_remain]
all_samples = np.concatenate([remaining_samples, added_samples])

words_transcriptions = {}
for word, transcription in zip(list(df_original['word']), list(df_original['transcription'])):
    words_transcriptions[word] = transcription

dropped = 0
added = 0
for sample in all_samples:

    try:
        if 'tokens' in sample:
            del sample['tokens']

        sample['transcription'] = words_transcriptions[sample['word']]
        added += 1
    except:
        dropped += 1

# Небольшой косяк, связанный с отличием версий исходного датасета
print(added)
print(dropped)

  
with open(ProcessedDataset.save_path, 'w', encoding='utf-8') as f:
    sep = PreparedDataset.sep
    titles = f"word{sep}transcription{sep}gender{sep}url{sep}path\n"
    f.write(titles)
    
    for sample in all_samples:
        try:
            f.write(f"{sample['word']}" + f"{sep}" + f"{sample['transcription']}" + f"{sep}" + f"{sample['gender']}" + f"{sep}" + f"{sample['url']}" + f"{sep}" + f"{sample['path']}\n")
        except:
            continue