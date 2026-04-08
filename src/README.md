# Tox21 Toxicity Prediction Pipeline

## Overview

This project implements an end-to-end machine learning pipeline for predicting molecular toxicity across multiple biological assays using the Tox21 dataset.

Each molecule is represented as a SMILES string and converted into numerical features using Morgan fingerprints. A separate Logistic Regression model is trained for each assay.

---

## Pipeline

### 1. Data Loading
- Load dataset from CSV
- Identify:
  - `smiles` (molecular structure)
  - `mol_id` (identifier)
  - assay columns (targets)
- Remove missing or empty SMILES
- Optionally remove duplicate molecules

---

### 2. Featurization
- Convert SMILES → Morgan fingerprints
- Fixed-length binary vectors (e.g. 2048 bits)

Output:
- `X`: feature matrix

---

### 3. Target Preparation
- Extract assay columns as labels
- Each assay is a separate binary classification task
- Missing labels are handled per assay during training

---

### 4. Train/Test Split
- Split dataset into training and test sets
- Same split used across all assays

---

### 5. Model Training
- Train one Logistic Regression model per assay
- For each assay:
  - remove rows with missing labels
  - train classifier on remaining data
- Class imbalance handled with `class_weight="balanced"`

---

### 6. Prediction
- Generate predictions on test set
- Outputs:
  - predicted class (0/1)
  - predicted probability

---

### 7. Evaluation

Per assay:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC

Aggregated:
- Macro average (across assays)
- Micro average (across all predictions)

---

### 8. Plots
- ROC curves per assay
- Precision-Recall curves per assay
- Metric bar charts (ROC-AUC, F1, PR-AUC)
- Class balance per assay

---

### 9. Error Analysis
- Build long-format prediction table:
  - molecule (SMILES, ID)
  - assay
  - true label
  - predicted label
  - predicted probability

Derived:
- misclassified samples
- most confident errors
- largest probability errors
- error summary per assay
- false positives / false negatives

---

### 10. Outputs

Results are saved to:
