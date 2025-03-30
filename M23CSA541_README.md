 SepID-Enhance: Speaker Separation, Identification, and Enhancement Pipeline

Introduction

This assignment explores the application of advanced speech processing techniques in complex audio environments, focusing on both multi-speaker speech enhancement and speaker verification, as well as language classification using MFCC features. The first part involves enhancing and separating speech signals in overlapping speaker scenarios using the SepFormer model. A pre-trained WavLM Base Plus model is selected for speaker verification, followed by fine-tuning using Low-Rank Adaptation (LoRA) and ArcFace loss on a subset of the VoxCeleb2 dataset. The system is evaluated using key metrics such as Equal Error Rate (EER), TAR@1%FAR, and Speaker Identification Accuracy. A novel pipeline is also designed to combine speaker separation, identification, and enhancement through joint training, further improving performance in real-world multi-speaker conditions.

The second part of the assignment focuses on language analysis through MFCC-based feature extraction from audio samples across 10 Indian languages. The acoustic properties of selected languages are explored through MFCC spectrograms, and differences are quantified via statistical analysis. These features are then used to build a language classification model, trained and tested on normalized MFCC vectors using a Random Forest classifier. The task emphasizes analyzing how well MFCCs capture language-specific characteristics and addresses challenges such as speaker variability and background noise.

 Overview

SepID-Enhance is a modular and extensible pipeline designed to tackle the complex challenges of multi-speaker audio processing. It unifies cutting-edge techniques in speech separation, speaker identification, and enhancement, along with language classification based on MFCC features. The system is ideal for tasks such as diarization, transcription, multilingual voice analytics, and audio scene understanding.

This project integrates:
- Speaker Separation  : using SepFormer
- Speaker Identification:- via WavLM (pre-trained + fine-tuned with LoRA & ArcFace)
- Speech Enhancement:- through joint training
- Language Classification: - using MFCCs and Random Forest

 Key Features

Speaker Separation & Enhancement
- Model: `speechbrain/sepformer-wsj02mix`
- Metrics:
  - SDR: 9.80
  - SIR: 10.50
  - SAR: 11.20
  - PESQ: 1.95

Speaker Identification
- Model: `microsoft/wavlm-base-plus`
- Pre-trained:
  - EER: 34.00%
  - TAR@1%FAR: 12.00%
  - Accuracy: 66.10%
- Fine-tuned:
  - EER: 52.48%
  - TAR@1%FAR: 0.29%
  - Accuracy: 47.40%
- Rank-1 Accuracy (on separated speech):
  - Pre-trained: 58.00%
  - Fine-tuned: 62.00%

Language Classification
- Dataset: 10 Indian Languages (Kaggle)
- Features: MFCCs
- Classifier: Random Forest
- Accuracy: 76.57%

Setup

 Datasets

	1:- VoxCeleb2
	2: -VoxCeleb1
	3: -Indian Languages Dataset: from Kaggle
 

Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install torch torchaudio transformers speechbrain pesq numpy tqdm peft librosa soundfile scikit-learn matplotlib
```

## GitHub
https://github.com/vivekpandey000023/Speech_Understanding_Assignment_2

