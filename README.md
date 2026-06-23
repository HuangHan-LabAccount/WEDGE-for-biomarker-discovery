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
pip install -r lib/requirements.txt
```
### 3. Proteomics Dataset

### Raw Proteomics Dataset
Due to the large file size of the raw mass spectrometry data, the complete dataset is hosted externally on the **iProX/ProteomeXchange** consortium:
* **Project ID:** `IPX0013995000` / `PXD074127`
* **Subproject ID:** `IPX0013995001`
* **Access Link:** [iProX Repository](https://www.iprox.cn/page/PSV023.html;?url=1782219624318Pdxl) 
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


---
The following scripts provide the complete implementation logic of the WEDGE framework as described in the manuscript. They are intended for methodological transparency and reference:

| Script Name | Operational Purpose | Key Deliverables & Visual Metrics |
| :--- | :--- | :--- |
| **`00_WGAN-GP_for_Data_Augmentation.py`** | Trains conditional WGAN-GP networks using raw data subsets to generate high-fidelity synthetic protein expression values for target clinical classes (`HPV_related` / `NHPV`). | Outputs balanced pseudo-expression CSV matrices saved under the `Aug_data/` directory. |
| **`00_Evaluation_of_Augdataset.py`** | Evaluates generative quality using **Fréchet Distance (FD Score)** compared to a statistical random baseline, alongside calculating 5-fold cross-validation accuracy metrics across the graph streams. | Generates FD score comparison charts (`fd_comparison.pdf`) and multi-stream accuracy bar plots (`accuracy_comparison.pdf`). |
| **`01_0_Train_and_Interpret_WEDGE.py`** | Implements the end-to-end 5-fold cross-validation training loop of the `GraphLevelHeteroGCN` network. Applies **Integrated Gradients (IG)** node explanation to isolate key clinical features. | Generates candidate diagnostic markers ranking metrics and exports custom-colored vector charts for Top PPI/GRN scores. |
| **`01_1_Visualization_of_WEDGE_loss(Train&Validation).py`** | Extracts and aggregates cross-validation training metrics from TensorBoard logs, employing **Exponential Moving Average (EMA) smoothing** to highlight foundational optimization patterns. | Exports multi-fold overlaid, publication-grade loss trend curves (`train_total_loss.pdf`, `val_total_loss.pdf`, etc.) into the figures module. |
| **`01_2_Melt_WGAN_GP_Experient.py`** | Performs **MELT experiments** tracking inter-sample Pearson correlation trends and maps *Intra-class* vs *Inter-class* convergence trajectories. Executes full-cycle ablation training without data augmentation. | Produces comparative clustermaps, localized correlation KDE distribution graphs, and comparative ablation confusion matrices. |
| **`02_Feature_Combination_Selection.py`** | Systematically reviews random feature blocks ranging across sizes from 3 to 10 nodes. Calculates recurrence distribution heatmaps and runs network-wide topology **Permutation Tests** utilizing `igraph`. | Generates combination score matrixes, occurrence frequency histograms, and null-distribution Z-score topology significance plots. |
| **`03_Evaluate_Internal_Test_Cohort.py`** | Tests the signature scale-dependence (from 1 to 15 biomarkers) across standard models (WEDGE, BINN, POC19, RF, DIABLO) over internal test and independent multicenter external cohorts. | Produces diagnostic signature accuracy lines, ROC curves, PRF multi-class metrics, and local **SHAP Feature Importance** plots. |

> **Note:** All scripts use **relative paths** based on the script location. Ensure the project directory structure remains unchanged when running locally.
Run the WEDGE framework directly with the demo dataset:

```bash
# Step 1: Run WGAN-GP data augmentation
python 00_WGAN-GP_for_Data_Augmentation.py

# Step 2: Evaluate data generation fidelity (FD metrics) and baseline accuracies
python 00_Evaluation_of_Augdataset.py

# Step 3: Train the WEDGE Heterogeneous GCN architecture and extract biomarker node importance scores 
python 01_0_Train_and_Interpret_WEDGE.py

# Step 4: Extract and visualize EMA-smoothed Loss curves from log events
python 01_1_Visualization_of_WEDGE_loss.py

# Step 5: Perform MELT correlation experiments and the non-augmented ablation study
python 01_2_Melt_WGAN_GP_Experient.py

# Step 6: Run combinatorial screening and topological network permutation testing
python 02_Feature_Combination_Selection.py

# Step 7: Complete benchmarking validations across internal and multi-center external cohorts
python 03_Evaluate_Internal_Test_Cohort.py
```

Results will be saved in the `output/` directory.



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
