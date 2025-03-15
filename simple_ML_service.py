import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from flask import Flask, request, jsonify

app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def audio_to_tensor(audio_file):

    batch = []

    waveform, sample_rate = torchaudio.load(audio_file.stream)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    if waveform.shape[0] > 1: # Если стерео-, квадро- и т.д., то усредняем
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    batch.append(waveform.squeeze())

    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

processor_whisper = WhisperProcessor.from_pretrained("openai/whisper-large")
processor_whisper.feature_extractor.return_attention_mask = True
model_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model_whisper.to(device)
decoder_ids = processor_whisper.get_decoder_prompt_ids(language="english", task="transcribe")

def Whisper_inference(waveforms):
    inputs = processor_whisper(waveforms.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=False)
    print(f"inputs shape: {inputs.input_features.shape}")
    print(f"attention_mask shape: {inputs.attention_mask.shape}\n")
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)
    predicted_ids = model_whisper.generate(input_features, forced_decoder_ids=decoder_ids, attention_mask=attention_mask)
    print(f"predicted_ids shape: {predicted_ids.shape}")

    return processor_whisper.batch_decode(predicted_ids, skip_special_tokens=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    try:
        audio_file = request.files['file']
        waveforms_correct = audio_to_tensor(audio_file)
        result = Whisper_inference(waveforms_correct)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    if result[0] == ' ':
        result = result[1:]
    if result[-1] in '.!?,':
        result = result[:-1]
    return jsonify({'transcription': result}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'OK!'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)