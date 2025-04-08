import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import os
import torchaudio

DATASET_PATH = "/home/speech-iitdh/Disk/Bhavana/Ankitha/wavlm_model/VoxCeleb1/vox1_dev_wav/wav"

for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                waveform, sample_rate = torchaudio.load(file_path)
                print(f"Loaded: {file_path} | Sample Rate: {sample_rate}")

