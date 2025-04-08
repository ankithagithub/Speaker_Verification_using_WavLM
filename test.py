import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, AutoProcessor
from sklearn.metrics import roc_curve, DetCurveDisplay
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/home/speech-iitdh/Disk/Bhavana/Ankitha/wavlm_model/wavlm_voxceleb_model"
TEST_DATASET_PATH = "/home/speech-iitdh/Disk/Bhavana/Ankitha/wavlm_model/VoxCeleb1/vox1_test_wav/wav"
NUM_TEST_PAIRS = 2000  # Increased test pairs for better EER estimation

# ================== LOAD MODEL ==================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = WavLMForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ================== TEST DATASET CLASS ==================
class VoxCelebTestDataset(Dataset):
    def __init__(self, dataset_path, processor, max_length=32000):
        self.dataset_path = dataset_path
        self.processor = processor
        self.max_length = max_length
        self.audio_files = []

        for speaker_id in os.listdir(dataset_path):
            speaker_path = os.path.join(dataset_path, speaker_id)
            if os.path.isdir(speaker_path):
                for session_id in os.listdir(speaker_path):
                    session_path = os.path.join(speaker_path, session_id)
                    if os.path.isdir(session_path):
                        for file in os.listdir(session_path):
                            if file.endswith(".wav"):
                                self.audio_files.append(os.path.join(session_path, file))

        print(f"Loaded {len(self.audio_files)} audio files from {dataset_path}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        return {"input_values": inputs.input_values.squeeze(0), "file_path": file_path}

# ================== LOAD TEST DATASET ==================
print("Loading test dataset...")
test_dataset = VoxCelebTestDataset(TEST_DATASET_PATH, processor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# ================== EXTRACT SPEAKER EMBEDDINGS ==================
def extract_embeddings(model, dataloader):
    embeddings = {}
    print("Extracting speaker embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_values = batch["input_values"].to(DEVICE)
            outputs = model(input_values, output_hidden_states=True)

            # Use the last hidden state
            hidden_states = outputs.hidden_states[-1]

            # Mean + Standard Deviation pooling
            mean_pooling = hidden_states.mean(dim=1)
            std_pooling = hidden_states.std(dim=1)
            embedding = torch.cat([mean_pooling, std_pooling], dim=1).cpu().numpy().flatten()

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[batch["file_path"][0]] = embedding

    print("Speaker embeddings extracted successfully!")
    return embeddings

speaker_embeddings = extract_embeddings(model, test_loader)

# ================== GENERATE TEST PAIRS ==================
def create_test_pairs(embeddings, num_pairs=NUM_TEST_PAIRS):
    files = list(embeddings.keys())
    same_speaker_pairs = []
    diff_speaker_pairs = []

    for _ in range(num_pairs // 2):
        spk1 = np.random.choice(files)
        same_speaker_candidates = [f for f in files if f.split('/')[-2] == spk1.split('/')[-2]]
        if len(same_speaker_candidates) > 1:
            spk1_same = np.random.choice(same_speaker_candidates)
            same_speaker_pairs.append((spk1, spk1_same, 1))
        
        spk2 = np.random.choice(files)
        diff_speaker_candidates = [f for f in files if f.split('/')[-2] != spk2.split('/')[-2]]
        if len(diff_speaker_candidates) > 1:
            spk2_diff = np.random.choice(diff_speaker_candidates)
            diff_speaker_pairs.append((spk2, spk2_diff, 0))

    print(f"Generated {len(same_speaker_pairs) + len(diff_speaker_pairs)} test pairs")
    return same_speaker_pairs + diff_speaker_pairs

test_pairs = create_test_pairs(speaker_embeddings)

# ================== COMPUTE SIMILARITY & GROUND TRUTH ==================
same_speaker_scores = []
diff_speaker_scores = []

print("Computing similarity scores...")
for file1, file2, label in tqdm(test_pairs):
    emb1 = torch.tensor(speaker_embeddings[file1])
    emb2 = torch.tensor(speaker_embeddings[file2])
    
    # Use torch cosine similarity
    score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    if label == 1:
        same_speaker_scores.append(score)
    else:
        diff_speaker_scores.append(score)

# ================== COMPUTE EER ==================
labels = np.concatenate([np.ones_like(same_speaker_scores), np.zeros_like(diff_speaker_scores)])
scores = np.concatenate([same_speaker_scores, diff_speaker_scores])
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr

# Compute EER using Brent’s method
eer_value = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.)

print(f"\n🔹 Improved Equal Error Rate (EER): {eer_value * 100:.2f}%")

# ================== PLOT ROC CURVE ==================
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.scatter(eer_value, 1 - eer_value, color='red', label=f"EER: {eer_value*100:.2f}%")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
print("ROC Curve saved as 'roc_curve_optimized.png'!")

