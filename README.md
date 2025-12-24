# Neural Network for Bird Noise Classification

This project implements a **convolutional neural network (CNN)** to classify bird species based on their vocalizations. Raw audio recordings are converted into **Mel spectrogram images**, augmented using audio-specific techniques, and used to train a deep learning model for multi-class bird species classification.

---

## Overview
- Converts bird vocalization audio files into Mel spectrogram images
- Applies **time masking and frequency masking** to improve model generalization
- Trains a custom CNN from scratch to classify bird species
- Evaluates performance using accuracy, precision, recall, F1-score, and confusion matrices

The project focuses on understanding how **data augmentation, model depth, and dataset size** affect performance in bioacoustic classification tasks.

---

## Dataset
- Source: **Cornell Birdcall Identification dataset [Kaggle](https://www.kaggle.com/competitions/birdsong-recognition/overview)**
- Subset of 5–10 bird species selected for experimentation
- Each species contains ~100 audio samples
- Audio files converted to spectrograms using `librosa`

---

## Methodology
1. **Audio Preprocessing**
   - Load MP3 files with `librosa`
   - Generate Mel spectrograms and convert to decibel scale
   - Resize spectrograms to 224×224 for CNN input

2. **Data Augmentation**
   - Time masking (horizontal masking in spectrograms)
   - Frequency masking (vertical masking in spectrograms)
   - Augmentation applied only to training data
   - Best results achieved with **50% augmented training samples**

3. **Model Architecture**
   - Custom CNN with 6 convolutional blocks
   - Increasing filter sizes (32 → 256)
   - ReLU activations and max pooling
   - Dropout layers for regularization
   - Dense layers for final classification

4. **Training Strategy**
   - Optimizer: Adam (learning rate = 0.001)
   - Loss: Sparse categorical cross-entropy
   - Early stopping based on validation loss
   - Train/validation/test split: 80% / 10% / 10% (stratified)

---

## Results
- **5 species:** up to **98% accuracy**
- **6 species:** up to **92% accuracy**
- Performance decreases as number of species increases due to limited dataset size
- Augmentation significantly improves generalization, but excessive augmentation degrades performance

Evaluation includes:
- Accuracy and F1-score
- Precision and recall per class
- Confusion matrices
- Training vs. validation accuracy and loss curves

---

## Tech Stack
- Python
- TensorFlow / Keras
- librosa
- tensorflow-io (audio augmentation)
- scikit-learn
- NumPy, Matplotlib, Seaborn

---

## Notes & Limitations
- Dataset size is small (~100 samples per class)
- Model trained on spectrogram images, not raw waveforms
- Results may not generalize to real-world field recordings with heavy background noise
- Project is intended as an educational deep learning pipeline, not a production system
