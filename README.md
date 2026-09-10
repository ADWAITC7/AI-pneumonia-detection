# Pneumonia Detection from Chest X-Rays

An end-to-end deep learning project for binary classification of chest X-ray images into:

- **NORMAL**
- **PNEUMONIA**

The project uses a pretrained **ResNet-18** model with transfer learning, a reproducible training/validation split, GPU acceleration with PyTorch, threshold-based evaluation, and a Streamlit web application for inference.

> **Disclaimer:** This project is for educational/research purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

---

## 🚀 Demo

The project includes a Streamlit interface where a user can upload a chest X-ray and receive a model prediction.

**App:** `app.py`

---

## 🧠 Approach

### Model

- Architecture: **ResNet-18**
- Framework: **PyTorch**
- Pretrained weights: ImageNet
- Final classifier: 2 output classes
- Classes:
  - `NORMAL = 0`
  - `PNEUMONIA = 1`

The pretrained CNN backbone provides useful visual features, while the final classification layer is adapted for the pneumonia detection task.

### Input preprocessing

Images are resized to:

`224 × 224`

The training pipeline uses augmentation such as:

- Random horizontal flip
- Random rotation

Validation and test preprocessing are deterministic so that evaluation is consistent.

Image normalization follows the ImageNet preprocessing convention used by the pretrained ResNet-18.

---

## 📊 Dataset

The project uses the Chest X-Ray Pneumonia dataset with the following structure:

```text
DATASET/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The physical validation folder contains only a small number of images, so the training pipeline creates a reproducible **80/20 train-validation split** from the training data using seed `42`.

Dataset counts used by the pipeline:

- Training source images: **5,216**
- Training split: **4,173**
- Validation split: **1,043**
- Test images: **624**

The training source is imbalanced toward the pneumonia class, which is important when interpreting model performance.

---

## ⚙️ Training

The project supports GPU training through CUDA when available.

Device selection follows:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The training pipeline tracks:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

The best checkpoint is saved based on validation performance.

Model files:

```text
model/
├── pneumonia_model.pth
└── pneumonia_model_finetuned.pth
```

The `.pth` files are intentionally excluded from normal Git tracking by `.gitignore`.

---

## 📈 Results

### Best validation result

The fine-tuned model reached:

**97.89% validation accuracy**

However, validation performance is not the same as final held-out test performance.

### Held-out test result

Using the fine-tuned model with a pneumonia probability threshold of **0.55**:

**Test Accuracy: 84.13%**

Confusion matrix:

```text
[[138, 96],
 [  3, 387]]
```

<img width="1882" height="906" alt="image" src="https://github.com/user-attachments/assets/3edf5748-2d88-4e9f-ba24-2bfa8dcfb742" />
<img width="1527" height="782" alt="image" src="https://github.com/user-attachments/assets/cd943166-99df-48f7-a1c8-36019e8b5183" />

Where:

- True Negatives (NORMAL correctly predicted): **138**
- False Positives (NORMAL predicted as PNEUMONIA): **96**
- False Negatives (PNEUMONIA predicted as NORMAL): **3**
- True Positives (PNEUMONIA correctly predicted): **387**

Key metrics:

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 0.9787 | 0.8012 |
| Recall | 0.5897 | 0.9923 |
| F1-score | 0.7360 | 0.8866 |

Overall:

- **Accuracy:** 84.13%
- **Pneumonia Recall (Sensitivity):** 99.23%
- **Normal Recall (Specificity):** 58.97%
- **False Positives:** 96
- **False Negatives:** 3

### What the results show

The model is highly sensitive to pneumonia, but it tends to over-predict pneumonia.

That means the current model produces:

- Very few missed pneumonia cases
- A relatively large number of false alarms on normal X-rays

This is an important example of why accuracy alone is not enough for medical-image classification.

The **0.55 threshold was selected using validation data**, not the held-out test set.

---

## 🔍 Error Analysis

The evaluation pipeline reports:

- Confusion matrix
- Classification report
- False positives
- False negatives
- High-confidence prediction errors

This makes it possible to inspect where the model fails instead of treating a single accuracy number as the whole story.

One important finding is the gap between validation and test performance. The fine-tuned model achieved very strong validation results but generalized less well to the held-out test set.

Possible contributors include dataset imbalance, distribution differences, and model decision bias. These are hypotheses rather than experimentally isolated causes in the current version.

---

## 🗂️ Project Structure

```text
pneumonia-detection/
│
├── DATASET/                      # Local dataset (not tracked by Git)
│
├── model/
│   ├── pneumonia_model.pth
│   └── pneumonia_model_finetuned.pth
│
├── src/
│   ├── data_loader.py            # Dataset loading and train/validation split
│   ├── model.py                  # ResNet-18 model definition
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Test evaluation and error analysis
│   ├── threshold_analysis.py     # Threshold comparison on validation data
│   └── test_data.py              # Data/model sanity checks
│
├── app.py                        # Streamlit inference application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pneumonia-detection
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

From the project root:

```powershell
streamlit run app.py
```

The app will open in your browser.

---

## 🧪 Run the Training Pipeline

```powershell
python src/train.py
```

The best checkpoint will be saved under:

```text
model/
```

---

## 📊 Evaluate the Model

```powershell
python src/evaluate.py
```

For validation threshold analysis:

```powershell
python src/threshold_analysis.py
```

---

## 💻 Hardware

The project was developed and tested with PyTorch CUDA acceleration.

The training code automatically falls back to CPU when CUDA is unavailable.

---

## 🔮 Possible Next Improvements

Potential future experiments include:

- Class-weighted loss or balanced sampling
- More systematic hyperparameter tuning
- Stronger augmentation strategies
- Early stopping and learning-rate scheduling
- External validation on another chest X-ray dataset
- Explainability methods such as Grad-CAM
- Calibration of prediction probabilities
- More careful analysis of dataset distribution and leakage

These should be treated as future experiments rather than claims about the current model.

---

## 📌 Key Learning Outcomes

This project demonstrates practical experience with:

- Convolutional neural networks
- Transfer learning
- ResNet architectures
- PyTorch training loops
- CUDA/GPU acceleration
- Image preprocessing and augmentation
- Train/validation/test methodology
- Confusion matrices and classification metrics
- Threshold selection
- Error analysis
- Streamlit deployment
- Reproducible ML workflows

---

## 📄 License

Add the license appropriate for your intended use before publishing the repository.
