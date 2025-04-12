import pandas as pd
import json

import warnings
warnings.filterwarnings('ignore')

unary_phonemes_list = [
    "æ","ɪ", "ɛ", "ɒ", "ʊ", "ʌ", "ə",
    "p", "f", "t", "θ", "s", "ʃ", "k", "b", "v", "d", "ð", "z", "ʒ", "ɡ", "h", "m", "n", "ŋ", "r", "l", "w", "j",
    "ˈ", "ˌ"
]

binary_phonemes_list = [
    "ɑː", "iː", "ɔː", "uː", "ɜː",
    "eɪ", "aɪ", "aʊ", "əʊ", "ɔɪ", "ɪə", "ɛə", "ʊə",
    "tʃ", "dʒ"
]

phonemes_list = unary_phonemes_list + binary_phonemes_list

def clean_transcription(transcription):
    return transcription[2:-2] # отсекаем слеши в начале и в конце

def drop_words_by_phoneme(phoneme, df):
    return df[~df["transcription"].str.contains(phoneme)]

def fix_transcription(word, fixed_transcription, df):
    df.loc[df["word"] == word, "transcription"] = fixed_transcription
    
def find_wrong_transcripts(df):
    transcriptions_list = list(df["transcription"])

    wrong_transcripts = []

    skip = False

    for transcription in transcriptions_list:
        for i in range(len(transcription)):
            
            if skip:
                skip = False
                continue
            
            if i < len(transcription) - 1:
                
                if transcription[i:i+2] in binary_phonemes_list:
                    skip = True
                
                elif transcription[i] in unary_phonemes_list:
                    pass
                    
                else:
                    wrong_transcripts.append(transcription)
                    
                continue
            
            if transcription[i] in unary_phonemes_list:
                pass
                    
            else:
                wrong_transcripts.append(transcription)
                
    return list(set(wrong_transcripts))

def get_phonemes_frequencies(df):
    
    assert(len(find_wrong_transcripts(df)) == 0)
    
    phonemes_dict = {phoneme: 0 for phoneme in phonemes_list}
    
    transcriptions_list = list(df["transcription"])

    skip = False

    for transcription in transcriptions_list:
        for i in range(len(transcription)):
            
            if skip:
                skip = False
                continue
            
            if i < len(transcription) - 1:
                
                if transcription[i:i+2] in binary_phonemes_list:
                    phonemes_dict[transcription[i:i+2]] += 1
                    skip = True
                
                elif transcription[i] in unary_phonemes_list:
                    phonemes_dict[transcription[i]] += 1
                    
                else:
                    wrong_transcripts.append(transcription)
                    
                continue
            
            if transcription[i] in unary_phonemes_list:
                phonemes_dict[transcription[i]] += 1
                    
            else:
                wrong_transcripts.append(transcription)
                
    phonemes = phonemes_dict.keys()
    frequencies = phonemes_dict.values()
                
    return phonemes, frequencies

def drop_word(word, df):
    return df[~(df["word"] == word)]

def show_word_by_transcription(transcription, df):
    return list(df[df["transcription"] == transcription]["word"])[0]


if __name__ == "__main__":
    
    df = pd.read_csv("../../data/external/labeled_dataset.txt", delimiter="№", 
                 header=None, names=[
                     "word",
                     "transcription",
                     "gender",
                     "url"
                 ])
    
    df_male = df[df.gender == "мужчина"]
    df_female = df[df.gender == "женщина"]
    
    df = pd.concat([df_male, df_female])
    
    df["transcription"] = df["transcription"].apply(clean_transcription)
    
    df_new = drop_words_by_phoneme("ø", df)
    df_new = drop_words_by_phoneme("y", df_new)
    
    fix_transcription("interplanetary", "ɪntəplænɪtərɪ", df_new)
    fix_transcription("lived", "lɪvd", df_new)
    
    df_new = drop_words_by_phoneme('̃', df_new)
    df_new = drop_words_by_phoneme("œ", df_new)
    df_new = drop_words_by_phoneme("x", df_new)
    df_new = drop_words_by_phoneme("o", df_new)
    
    wrong_transcripts = find_wrong_transcripts(df_new)
    
    wrong_words = {}
    for transcription in set(wrong_transcripts):
        wrong_words[show_word_by_transcription(transcription, df_new)] = transcription
    
    for word in wrong_words:
        df_new = drop_word(word, df_new)
        
    df_new.to_csv("../../data/prepared/prepared_dataset.csv", sep="№")
    
    phonemes, frequencies = get_phonemes_frequencies(df_new)
    frequencies_sorted, phonemes_sorted = zip(*sorted(zip(frequencies, phonemes), reverse=True))
    
    phonemes_vocab = {"<pad>": 0, "<unk>": 1}
    i = 2

    for phoneme in phonemes_sorted:
        
        phonemes_vocab[phoneme] = i
        i += 1
        
    with open("../../data/prepared/phonemes_vocab.json", "w") as json_file:
        json.dump(phonemes_vocab, json_file)

    
