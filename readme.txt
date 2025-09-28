# SQL Injection Detection Using Machine Learning

**This project is based on the following paper:**  
**Title:** SQL Injection Detection Using Machine Learning Methods  
**Authors:** Mahboobe Sarabinejad, Zahra Eskandari  
**Conference:** 7th International Conference on Internet of Things and Applications  
**National Scientific Document ID:** IOTCONF07_012  
**Indexing Date:** January 7, 2024 (18 Dey 1402)  
**Link:** https://civilica.com/doc/1878981/

---

## Project Overview

The goal of this project is **detecting SQL Injection (SQLi) attacks** by extracting syntactic and statistical features from SQL statements and training an **MLP (Multi-Layer Perceptron)** model. The input is a SQL query, and the output indicates whether the query is malicious or normal.

---

## Dataset

- Original dataset: `SQLiV3.csv` (~30.9k rows)  
- Processed dataset: `FinalDataset.csv` (~30.6k rows after removing NaN)

Columns:

- `Sentence` — SQL statement  
- `Label` — 1 = SQLi, 0 = normal  
- `LP` — Length of the statement  
- `NSPA` — Number of spaces  
- `RSPA` — Ratio of spaces to length  
- `NSPE` — Number of special characters  
- `RSPE` — Ratio of special characters to length  
- `NK` — Number of SQL keywords  
- `KWS` — Weighted sum of keywords  
- `ROC` — Composite index: 1 − (RSPA + RSPE)

---

## Feature Extraction

Steps implemented in `Dataset.ipynb`:

1. Remove unnecessary columns from the raw CSV.  
2. Compute length, spaces, and special characters.  
3. Count SQL keywords (e.g., `select`, `union`, `load_file`).  
4. Assign weights to keywords (KWS).  
5. Compute ratios and composite index (ROC).  
6. Drop incomplete rows and save to `FinalDataset.csv`.

---

## Model (MLP)

**This project uses only an MLP (Multi-Layer Perceptron) model**. Details:

- Input layer: Dense, `input_dim = 8` (corresponding to 8 numeric features)  
- Hidden layers: Several Dense layers (e.g., 4 layers with 100 nodes) with `relu` activation  
- Output layer: Dense with 1 node, `sigmoid` activation (binary classification)  
- Loss function: `binary_crossentropy`  
- Optimizer: `Adam` (learning_rate = 0.001 recommended)  
- Evaluation: Accuracy and confusion matrix  
- Training: K-Fold cross-validation or `train_test_split`  

The model is saved as `model.h5` (Keras) and/or `model.pkl` (pickle). Feature names are in `features.pkl`.

---

## Inference Example

```python
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("exports/model.h5")

# Load feature names
with open("exports/features.pkl", "rb") as f:
    features = pickle.load(f)

# Example function to extract features from a SQL statement
# def extract_features_from_sentence(sentence) -> [LP, NSPA, RSPA, NSPE, RSPE, NK, KWS, ROC]

sentence = "SELECT * FROM users WHERE id = 1 OR 1=1"
X_input = np.array([extract_features_from_sentence(sentence)])

prob = model.predict(X_input)[0][0]
label = 1 if prob >= 0.5 else 0
print(f"Probability: {prob:.4f} — Label: {label} ({'SQLi' if label==1 else 'Normal'})")
