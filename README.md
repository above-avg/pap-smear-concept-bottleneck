# Unsupervised Concept Bottleneck for Pap Smear Cytology

## Overview

This project explores an **unsupervised concept bottleneck framework for cytology image classification** using Pap smear microscopy images.

Instead of training a fully end-to-end classifier, the pipeline learns **interpretable morphological concepts** from cytology patches using **self-supervised learning (DINO)** and then aggregates these concepts for slide-level prediction.

The goal is to create representations that are:

* biologically meaningful
* robust to dataset size
* interpretable for medical analysis

---

## Dataset

This project uses the **CERVIX93 Pap Smear Dataset**.

Dataset characteristics:

| Category     | Count |
| ------------ | ----- |
| Negative     | 16    |
| LSIL         | 46    |
| HSIL         | 31    |
| Total slides | 93    |

Each case contains:

* microscopy image frames
* Extended Depth of Field (EDF) images
* manually annotated cell points

Total annotated cells: **2705**

| Class    | Annotated Cells |
| -------- | --------------- |
| Negative | 238             |
| LSIL     | 1536            |
| HSIL     | 931             |

---

## Pipeline

The system follows a multi-stage pipeline:

```
Cytology Frames
      │
      ▼
Patch Extraction (224×224)
      │
      ▼
Self-Supervised Training (DINO ViT)
      │
      ▼
Patch Embedding Extraction
      │
      ▼
K-Means Concept Discovery
      │
      ▼
Slide-Level Concept Representation
      │
      ▼
Classifier (MLP)
```

This approach transforms raw image patches into **interpretable concept clusters**.

---

## Methodology

### 1. Patch Extraction

Each EDF frame is divided into patches:

* Patch size: **224 × 224**
* Overlapping stride for better coverage
* Background filtering to remove empty regions

Output: a patch dataset used for representation learning.

---

### 2. Self-Supervised Representation Learning

We use **DINO (Distillation with No Labels)** with a Vision Transformer backbone.

Key properties:

* no manual annotations required
* learns visual patterns through teacher-student training
* captures fine morphological features

The model learns representations sensitive to:

* nuclear size
* chromatin texture
* cytoplasm patterns
* cell density
* staining variations

---

### 3. Embedding Extraction

After training, the encoder converts each patch into a **feature vector**.

Example:

```
patch → encoder → 384-dimensional embedding
```

These embeddings describe morphological patterns in the data.

---

### 4. Concept Discovery via Clustering

K-means clustering groups similar embeddings:

```
Embeddings → K-means (K ≈ 30) → Concept clusters
```

Clusters often correspond to cytological structures such as:

* mature squamous cells
* koilocytosis
* hyperchromatic nuclei
* inflammatory cells
* background debris

---

### 5. Slide-Level Representation

Each slide is represented by **cluster frequency vectors**:

```
Slide representation = histogram of concept clusters
```

This creates a **concept bottleneck layer** between image data and classification.

---

### 6. Classification

A shallow model (MLP or logistic regression) predicts the cytology grade:

```
Negative
LSIL
HSIL
```

based on the discovered concept distribution.

---

## Project Structure

```
pap_wsi_project/
│
├── src/
│   ├── extract_patches.py
│   ├── flatten_patches.py
│
├── notebooks/
│
├── dino/                 # external repo (not tracked)
│
├── data/                 # dataset (not tracked)
│
├── patches/              # generated patches (not tracked)
│
├── ssl_dataset/          # training patches (not tracked)
│
├── models/               # trained models (not tracked)
│
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/above-avg/pap-smear-concept-bottleneck.git
cd pap-smear-concept-bottleneck
```

Create a Python environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install torch torchvision timm scikit-learn opencv-python tqdm
```

Clone the DINO repository:

```
git clone https://github.com/facebookresearch/dino.git
```

---

## Running the Pipeline

### Extract patches

```
python src/extract_patches.py
```

### Prepare dataset for SSL

```
python src/flatten_patches.py
```

### Train DINO

```
python dino/main_dino.py \
--arch vit_small \
--data_path ssl_dataset \
--epochs 200
```

---

## Research Motivation

Traditional cytology classifiers often behave as **black-box CNN models**.

This project explores whether:

* morphological concepts can emerge **without labels**
* clustering in representation space aligns with **pathological features**
* concept distributions can explain disease grades

---

## Future Work

Possible extensions include:

* concept attribution analysis
* cluster visualization and expert validation
* multiple instance learning (MIL)
* causal concept intervention
* counterfactual cluster manipulation

---

## Acknowledgements

* CERVIX93 dataset authors
* Facebook Research for DINO
* PyTorch ecosystem

---

## License

This project is intended for **research and educational purposes**.
