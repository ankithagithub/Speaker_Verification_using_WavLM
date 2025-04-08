import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, AutoProcessor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== MODEL & PROCESSOR ==================
MODEL_NAME = "/home/speech-iitdh/Disk/Bhavana/Ankitha/wavlm_model/wavlm-large"
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# ================== DATASET CLASS ==================
class VoxCelebDataset(Dataset):
    def __init__(self, dataset_path, processor, max_length=32000):
        self.dataset_path = dataset_path
        self.processor = processor
        self.max_length = max_length
        self.audio_files = []
        self.labels = {}
        self.speaker_to_label = {}
        speaker_ids = set()

        # Scan dataset directory
        for speaker_id in os.listdir(dataset_path):
            speaker_path = os.path.join(dataset_path, speaker_id)
            if os.path.isdir(speaker_path):
                speaker_ids.add(speaker_id)

        # Assign labels from 0 to num_labels-1
        self.speaker_to_label = {speaker_id: i for i, speaker_id in enumerate(sorted(speaker_ids))}
        
        # Load files
        for speaker_id in self.speaker_to_label:
            speaker_path = os.path.join(dataset_path, speaker_id)
            for session_id in os.listdir(speaker_path):
                session_path = os.path.join(speaker_path, session_id)
                if os.path.isdir(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith(".wav"):
                            file_path = os.path.join(session_path, file)
                            self.audio_files.append(file_path)
                            self.labels[file_path] = self.speaker_to_label[speaker_id]

        self.num_labels = len(self.speaker_to_label)
        print(f"Total unique speakers: {self.num_labels}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[file_path]

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Convert to single-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Process input
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ================== INITIALIZE DATASET & MODEL ==================
DATASET_PATH = "/home/speech-iitdh/Disk/Bhavana/Ankitha/wavlm_model/VoxCeleb1/vox1_dev_wav/wav"
train_dataset = VoxCelebDataset(DATASET_PATH, processor)

# Load model with correct number of labels
model = WavLMForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=train_dataset.num_labels).to(DEVICE)

# ================== COLLATE FUNCTION ==================
def collate_fn(batch):
    input_values = [d["input_values"] for d in batch]
    labels = torch.tensor([d["labels"] for d in batch])

    # Pad sequences
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0)

    return {
        "input_values": input_values.to(DEVICE),
        "labels": labels.to(DEVICE)
    }

# ================== DATALOADER ==================
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ================== OPTIMIZER & LOSS FUNCTION ==================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Enable mixed precision training (only if CUDA is available)
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

# ================== TRAINING LOOP ==================
EPOCHS = 20
gradient_accumulation_steps = 4

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for i, batch in enumerate(loop):
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(input_values=batch["input_values"]).logits
            loss = loss_fn(outputs, batch["labels"]) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss/len(train_loader):.4f}")

# ================== SAVE TRAINED MODEL ==================
model.save_pretrained("./wavlm_voxceleb_model")
print("Model training complete & saved.")
