# WavLM-Large Speaker Recognition on VoxCeleb1

This repository contains code to train a **WavLM-Large** model for **speaker recognition** using the **VoxCeleb1** dataset. The model is fine-tuned using `WavLMForSequenceClassification` and achieves an **Equal Error Rate (EER) of 4.60%**.

---

## 📁 Project Structure
├── train.py                
├── test.py                 
├── VoxCelebDataset         
├── wavlm_voxceleb_model/   
└── README.md              

---

## 🔧 Environment Setup

Create a virtual environment and install required packages:

```bash
# Create and activate conda environment
conda create -n wavlm_env python=3.8 -y
conda activate wavlm_env

# Install PyTorch (for CUDA 11.8, adjust if needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace Transformers and utilities
pip install transformers datasets tqdm
```
💡 Make sure CUDA is properly installed if training on GPU.

🎧 Dataset: VoxCeleb1
Download the VoxCeleb1 development set from here and organize it like this:

```bash
wavlm-voxceleb/
├── VoxCeleb1/
│   └── vox1_dev_wav/
│       └── wav/
│           ├── id10001/
│           │   ├── 1zcIwhmdeo4/
│           │   │   ├── 00001.wav
│           │   │   └── ...
│           │   └── ...
│           └── ...

```
Update this line in train.py:

```bash
DATASET_PATH = "/path/to/vox1_dev_wav/wav"
```
🚀 Training the Model
To start training, run:

```bash
python train.py
```
Uses WavLM-Large from HuggingFace

Trains with CrossEntropyLoss

Model is saved to ./wavlm_voxceleb_model

🧪 Testing the Model
Run the test script:

```bash
python test.py
```
Evaluates the trained model

Calculates accuracy and Equal Error Rate (EER)

Modify test.py if needed for speaker verification tasks

📊 Results
Model	Dataset	EER (%)
WavLM-Large	VoxCeleb1	4.60

🧠 Model Details
Model: WavLM-Large

Loss: CrossEntropyLoss (speaker classification)

Batch Size: 4

Gradient Accumulation: 4 steps

Optimizer: AdamW (lr=5e-5)

Epochs: 20

Mixed Precision: Enabled via torch.cuda.amp

🧠 Future Work
Replace classification head with AM-Softmax or Cosine Similarity loss for better EER

Add evaluation script with scoring backend (e.g., cosine scoring or PLDA)

Fine-tune on other datasets (e.g., I-MSV, VoxCeleb2)

Explore domain adaptation for cross-lingual speaker recognition
