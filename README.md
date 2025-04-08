WavLM-Large Speaker Recognition on VoxCeleb1

This repository contains code to train a **WavLM-Large** model for **speaker recognition** using the **VoxCeleb1** dataset. The model is fine-tuned for classification using `WavLMForSequenceClassification` and achieves an **Equal Error Rate (EER) of 4.60%**.

## 📁 Project Structure

├── train.py # Main training script 
├── VoxCelebDataset # Custom PyTorch dataset class 
├── wavlm_voxceleb_model/ # Saved fine-tuned model 
└── README.md # You're here!


## 🔧 Environment Setup

This project uses **Python 3.8+** and **PyTorch** with **HuggingFace Transformers**. Create a virtual environment and install dependencies:


# Create conda environment
conda create -n wavlm_env python=3.8 -y
conda activate wavlm_env

# Install PyTorch (adapt for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace Transformers and other tools
pip install transformers datasets tqdm
🎧 Dataset: VoxCeleb1
Download the VoxCeleb1 Development Set:

Register at http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

Extract audio to the following path structure:

bash
Copy
Edit
/path/to/vox1_dev_wav/wav/
└── speaker_id/
    └── session_id/
        └── audio.wav
Update the DATASET_PATH variable in the script:

python
Copy
Edit
DATASET_PATH = "/path/to/vox1_dev_wav/wav"
🚀 Training the Model
Run the training script:

bash
Copy
Edit
python train.py
Uses WavLMForSequenceClassification from HuggingFace

Gradient accumulation & mixed precision enabled

Model checkpoint saved to ./wavlm_voxceleb_model

🧪 Performance
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

📬 Contact
If you use this code or have questions, feel free to open an issue or reach out!

