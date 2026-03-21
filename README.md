# WEDGE: Deep Learning-Enabled Proteomic Profiling for GAS Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **WEDGE**, a deep learning framework for the proteomic diagnosis of Gastric-type Adenocarcinoma of the Uterine Cervix (GAS).

---

## Abstract

Gastric-type adenocarcinoma of the uterine cervix (GAS) is a rare, aggressive, HPV–independent malignancy. Diagnostic identification of GAS is challenged by its potential mimicry of benign cervical glands and morphological overlap with HPV-associated adenocarcinoma. Such ambiguities can lead to clinical misclassification, necessitating objective biomarkers to ensure accurate diagnosis. 

Here, we present the first comprehensive, multi-center proteomic analysis of GAS, profiling 407 cervical tissue samples to define its molecular landscape. To identify robust biomarkers, we developed **WEDGE**, a deep learning framework synergizing Wasserstein Generative Adversarial Networks (WGAN-GP) for data augmentation with biologically informed Dual-stream Graph Convolutional Networks (GCN). WEDGE identified a robust two-protein diagnostic panel, **Pepsinogen C (PGC)** and **DNA Methyltransferase 1 (DNMT1)**, which distinguished GAS from HPV-CA with **93% accuracy** in the test cohort and **97% accuracy** in an independent external validation cohort, significantly outperforming existing biomarker discovery methods. Immunohistochemical validation of PGC and DNMT1 confirmed expression patterns consistent with our proteomic findings and high diagnostic accuracy. Notably, we established PGC as an independent prognostic factor and integration of PGC with clinicopathological variables yielded a risk stratification model that significantly outperformed conventional clinical models in predicting patient outcomes. Collectively, our study establishes a novel AI-driven proteomic framework for biomarker discovery and provides clinically actionable diagnostic and prognostic biomarkers in GAS.

---

## Model Architecture

![WEDGE Architecture](Fig1.png)
---

## System Requirements

### Hardware
* **CPU:** Standard desktop computer (minimum 16GB RAM recommended).
* **GPU:** NVIDIA GPU with 8GB+ VRAM (Optional, for faster inference/training).

### Software
* Python 3.8+
* **Dependencies:** (See `requirements.txt` for details)
    * `torch==2.1.0`
    * `pytorch-lightning==2.4.0`
    * `torch-geometric==2.6.1`  
---


## Getting Started

To use the WEDGE framework, clone the repository and install the necessary dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/HuangHan-LabAccount/WEDGE-for-biomarker-discovery.git
cd WEDGE-for-biomarker-discovery
```
### 2. Environment Setup
```bash
conda create -n wedge_env python=3.8
conda activate wedge_env
pip install -r requirements.txt
```
### 3. Proteomics Dataset

Due to the large file size of the raw mass spectrometry data, the complete dataset is hosted externally on the **iProX/ProteomeXchange** consortium:
### Raw Proteomics Dataset
Due to the large file size of the raw mass spectrometry data, the complete dataset is hosted externally on the **iProX/ProteomeXchange** consortium:
* **Project ID:** `IPX0013995000` / `PXD074127`
* **Subproject ID:** `IPX0013995001`
* **Access Link:** [iProX Repository](https://www.iprox.cn/page/PSV023.html;?url=1774021395032q4nj) 
* **Note:** Passwords for reviewer access are provided in the *Reporting Summary* submitted with the manuscript.

### Processed Data (Provided for Peer Review)
To facilitate the peer-review process and ensure reproducibility, the processed protein expression matrices and metadata are provided as a supplementary attachment (`data_for_WEDGE.zip`). 
* **Internal Cohort (Train & Test):** `expr_selected.csv` (protein expression matrix) and `meta_selected.csv` (metadata).
* **External Validation Cohort:** `expr_external_GAS_HPVCA.csv` and `meta_external_GAS_HPVCA.csv`. These represent a completely independent, multi-center dataset used for rigorous external validation.
---
### 4. Core Scripts

The WEDGE framework first employs **WGAN-GP** (Wasserstein Generative Adversarial Network with Gradient Penalty) for data augmentation of patient proteomic samples to address data scarcity. Subsequently, it utilizes a dual-stream Heterogeneous GCN to integrate biological priors from:
1.  **Protein-Protein Interaction (PPI) stream** (String database).
2.  **Gene Regulatory Network (GRN) stream** (TRRUST database).

The following scripts provide the complete implementation logic of the WEDGE framework as described in the manuscript. They are intended for methodological transparency and reference:
---

### `01_Train_and_Interpret_WEDGE.py`
* This is the core engine of our framework. It documents the construction and training of the dual-stream Heterogeneous GCN, integrating Protein-Protein Interaction (PPI) and Gene Regulatory Network (GRN) biological priors. Crucially, it includes the node importance explainer (using Integrated Gradients) which successfully identifies **PGC** and **DNMT1** as the top diagnostic biomarkers.

### `02_Evaluate_Internal_Test_Cohort.py`
* This script conducts a comprehensive evaluation on the internal test set. It dynamically assesses diagnostic accuracy across varying numbers of protein signatures and rigorously compares WEDGE against established baseline machine learning models (e.g., Random Forest, DIABLO, BINN, POC19). It also incorporates SHAP (SHapley Additive exPlanations) analysis to interpret the decision-making contribution of the identified biomarkers.

### `03_Evaluate_External_Validation_Cohort.py`
* This script demonstrates the clinical generalizability of our findings. It evaluates the fixed WEDGE signature (PGC & DNMT1) and baseline models purely on the **independent external multi-center cohort**. It generates key clinical diagnostic metrics, including external ROC curves, PRF (Precision, Recall, F1) scores, and confusion matrices.
---
### Model Evaluation (`Evaluation_of_WEDGE.py`)
* **Usage:** Orchestrates data loading, model initialization, and testing across different folds and cohorts.
* **Expected Output:** The script is designed to output a detailed evaluation matrix including **Accuracy**, **AUC**, **Sensitivity**, and **Specificity** for the GAS vs. HPV-CA classification task across internal and external validation cohorts.

### Model Training & Checkpoints (`Train.py`)
* **Usage:** A wrapper for the PyTorch Lightning trainer. It orchestrates the end-to-end training of the dual-stream Heterogeneous GCN, managing callbacks such as `ModelCheckpoint` and `EarlyStopping`.
* **Expected Output:** Automated logging of real-time metrics (Loss, Accuracy) and saving of the best model weights (`.ckpt`) based on validation loss into the `lib/checkpoints/` directory.

### Model Interpretability (`WEDGE_Explain.py`)
* **Usage:** Implements the interpretability layer using **Integrated Gradients (IG)** and **Grad-CAM** to rank proteins based on their diagnostic contribution.
* **Expected Output:** Generates node importance scores, supporting the biological findings and identifying PGC and DNMT1 as the top-ranking diagnostic biomarkers.
---

### Data Engineering & Augmentation
* **`utilsdata.py`**: Contains the complete data pipeline, including matrix normalization, data splitting, and graph construction. Key functions include `getAdjByString` for adjacency matrix generation and `build_hetero_graph_dataset` for creating `HeteroData` objects.
* **`WGAN-GP/`**: A dedicated module for Wasserstein Generative Adversarial Networks with Gradient Penalty. This is used for clinical data augmentation to address the scarcity of GAS samples.
* **`PPI_GRN_database/`**: This directory stores the biological prior knowledge databases (e.g., STRING and TRRUST) used to construct the dual-stream graph networks.

### Supporting Modules
* **`lib/checkpoints/`**: Storage for pre-trained model weights (`.ckpt`). The provided checkpoint allows for rapid reproduction of the results reported in the study.
* **`Compare/`**: Includes benchmark implementations of other biomarker discovery methods for performance comparison.
