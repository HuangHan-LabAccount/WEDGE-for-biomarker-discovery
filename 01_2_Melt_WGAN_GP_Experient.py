#%% ============================================================
# MELT Experiment: Compare WEDGE (with Augmentation) vs Ablation (No Augmentation)
# ============================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch
from tqdm import tqdm
sys.path.append('lib/')
from utilsdata import *
from Train import *
from WEDGE_model import GraphLevelHeteroGCN
from WEDGE_Explain import HGCN_Node_Importance_Explianer
from torch_geometric.data import DataLoader
# Project paths
plt.rcParams['pdf.fonttype'] = 42
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(os.path.join(PROJECT_ROOT, 'lib/'))



# ----- Data & Output Paths -----
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
ABLATION_CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints_ablation')

SAVE_DIR = os.path.join(OUTPUT_DIR, 'MELT')
os.makedirs(SAVE_DIR, exist_ok=True)

def apply_academic_axis(ax):
    """学术风格坐标轴"""
    ax.tick_params(axis='both', which='major', direction='out',
                   top=False, right=False, width=1.5, length=6, labelsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# PART 1: MELT Similarity Heatmap (Original vs Augmented)
# ============================================================
# --- Load Data ---
if True:
    print("=" * 60)
    print("PART 1: MELT Similarity Heatmap")
    print("=" * 60)
    meta = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_selected.csv'), index_col=0)
    matrix = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T
    matrix_Degene_sub = DEgene_selected(matrix, path=PROJECT_ROOT)
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    meta_sub = meta_sub.copy()
    meta_sub['CancerType'] = meta_sub['CancerType'].replace({'HPV_related': 'HPV_CA', 'NHPV': 'GAS'})
    matrix_sub = matrix_Degene_sub.loc[meta_sub.MS_number, :]

    # Original train/test split
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    custom_mapping = {'HPV_CA': 0, 'GAS': 1}
    mapped_train_labels = meta_train.CancerType.map(custom_mapping).values
    label_train = torch.tensor(mapped_train_labels, dtype=torch.float)
    mapped_test_labels = meta_test.CancerType.map(custom_mapping).values
    label_test = torch.tensor(mapped_test_labels, dtype=torch.float)

    # PPI and GRN networks
    protein_matrix_PPI = load_Stringdatabase(
        path=os.path.join(PPI_DIR, 'String_database/'),
        file_name="human_PPI_score_Stringdatabase(700up).csv")
    protein_matrix_GRN = load_Stringdatabase(
        path=os.path.join(PPI_DIR, 'Trrust_database/'),
        file_name="TF_filtered_human.csv")
    adj_PPI = getAdjByString(protein_matrix_PPI, matrix_train, one_direction=False)
    adj_GRN = getAdjByString(protein_matrix_GRN, matrix_train, one_direction=True)

    # Best WEDGE model is from fold 5 -> use fold 5 augmented data to match
    BEST_FOLD = 5
    AUG_FOLD = BEST_FOLD  # align augmented data fold with best model fold

    # Load augmented data for the corresponding fold
    aug_label0 = pd.read_csv(os.path.join(AUG_DATA_DIR, f'generated_data_fold{AUG_FOLD}_0.csv'))
    aug_label0.index = aug_label0['id']
    aug_label1 = pd.read_csv(os.path.join(AUG_DATA_DIR, f'generated_data_fold{AUG_FOLD}_1.csv'))
    aug_label1.index = aug_label1['id']
    aug_label0 = aug_label0.drop(columns=['subset', 'label', 'id'])
    aug_label1 = aug_label1.drop(columns=['subset', 'label', 'id'])

    print(f"Augmented data fold {AUG_FOLD}: label0={len(aug_label0)}, label1={len(aug_label1)}")

    # Label mapping
    label_map = {1.0: 'GAS', 0.0: 'HPV_CA'}
    display_labels_list = ['HPV_CA', 'GAS']
    lut = {'HPV_CA': "#E64B35", 'GAS': "#3C5488"}
    # ============================================================
    # Compute similarity on TRAINING SET ONLY (no test leakage)
    # ============================================================
    print("Computing similarity on training data only...")

    # --- Original training data similarity ---
    y_train_series = pd.Series(
        [label_map[float(v)] for v in label_train.numpy()],
        index=matrix_train.index,
        name='CancerType'
    )
    sim_matrix_original = matrix_train.T.corr(method='pearson')

    # --- Augmented training data similarity ---
    X_aug_train = pd.concat([matrix_train, aug_label0, aug_label1], axis=0)
    X_aug_train_reset = X_aug_train.reset_index(drop=True)
    y_gen0 = ['HPV_CA'] * len(aug_label0)
    y_gen1 = ['GAS'] * len(aug_label1)
    y_aug_train_list = list(y_train_series) + y_gen0 + y_gen1
    y_aug_train_series = pd.Series(y_aug_train_list, index=X_aug_train_reset.index, name='CancerType')
    sim_matrix_augmented = X_aug_train_reset.T.corr(method='pearson')

    print(f"Original training samples: {matrix_train.shape[0]}")
    print(f"Augmented training samples: {X_aug_train.shape[0]}")
# ============================================================
# Plotting function
# ============================================================
def plot_similarity_heatmap(sim_matrix, label_series, title, filename):
    plt.figure()
    row_colors = label_series.map(lut)
    g = sns.clustermap(
        sim_matrix,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="RdBu_r",
        center=0,
        metric='euclidean',
        method='average',
        figsize=(10, 10),
        dendrogram_ratio=(.15, .15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=False,
        yticklabels=False
    )
    g.fig.suptitle(title, fontsize=20, y=0.98)
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut.keys(), title='CancerType',
               bbox_to_anchor=(0.02, 0.75), bbox_transform=plt.gcf().transFigure,
               loc='upper left', fontsize=12)
    full_path = os.path.join(SAVE_DIR, f"{filename}.pdf")
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {full_path}")
    plt.show()
    plt.close()

# Save correlation matrices
sim_matrix_original.to_csv(os.path.join(SAVE_DIR, 'sim_matrix_original.csv'))
sim_matrix_augmented.to_csv(os.path.join(SAVE_DIR, 'sim_matrix_augmented.csv'))

plot_similarity_heatmap(
    sim_matrix_original,
    y_train_series,
    title="Inter-sample Similarity (Original Training Data)",
    filename="Similarity_Original_Training"
)

plot_similarity_heatmap(
    sim_matrix_augmented,
    y_aug_train_series,
    title=f"Inter-sample Similarity (Augmented Training Data, Fold {AUG_FOLD})",
    filename="Similarity_Augmented_Training"
)
# ============================================================
# PART 1.5: Intra-class vs Inter-class Correlation Analysis
# ============================================================
print("\n" + "=" * 60)
print("PART 1.5: Intra-class vs Inter-class Correlation Analysis")
print("=" * 60)

def compute_intra_inter_metrics(sim_matrix, label_series):
    """
    Compute intra-class and inter-class correlation statistics.

    sim_matrix: (n_samples x n_samples) Pearson correlation matrix
    label_series: pd.Series with sample labels aligned to sim_matrix rows/cols

    Returns dict with:
      - intra_GAS: list of all pairwise correlations within GAS class
      - intra_HPV: list of all pairwise correlations within HPV_CA class
      - inter: list of all pairwise correlations between different classes
    """
    labels = label_series.values
    n = len(labels)

    # Compute pairwise upper-triangle correlations by class
    intra_gas, intra_hpv, inter = [], [], []

    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only (no self-correlation)
            corr = float(sim_matrix.iloc[i, j])
            label_i = str(labels[i])
            label_j = str(labels[j])

            if label_i == 'GAS' and label_j == 'GAS':
                intra_gas.append(corr)
            elif label_i == 'HPV_CA' and label_j == 'HPV_CA':
                intra_hpv.append(corr)
            else:  # different classes
                inter.append(corr)

    return {
        'intra_GAS': np.array(intra_gas),
        'intra_HPV_CA': np.array(intra_hpv),
        'inter_GAS_HPVCA': np.array(inter)
    }


def summarize_intra_inter(stats_orig, stats_aug):

    """Print summary table of intra/inter metrics (no statistical testing)."""

    rows = []
    keys = [('Intra-GAS', 'intra_GAS'), ('Intra-HPV_CA', 'intra_HPV_CA'),
            ('Inter (GAS VS HPV_CA)', 'inter_GAS_HPVCA')]

    for name, key in keys:
        s_orig = stats_orig[key]
        s_aug  = stats_aug[key]
        orig_mean, orig_std = np.mean(s_orig), np.std(s_orig)
        aug_mean,  aug_std  = np.mean(s_aug),  np.std(s_aug)
        delta = aug_mean - orig_mean

        rows.append([name, orig_mean, orig_std, aug_mean, aug_std, delta])

    df_summary = pd.DataFrame(rows, columns=['Metric', 'Original Mean', 'Original Std', 'Augmented Mean', 'Augmented Std', 'Delta'])

    print("\n--- Intra/Inter Correlation Summary ---")
    print(df_summary.to_string(index=False))

    csv1 = os.path.join(SAVE_DIR, 'intra_inter_correlation_summary.csv')
    df_summary.to_csv(csv1, index=False)
    print(f"Saved: {csv1}\n")
    return df_summary


def add_significance_bracket(ax, x1, x2, y, p, h=0.03):
    """Draw a significance bracket between two x positions."""
    if p < 0.001:
        label, y_text = '***', y + h * 3
    elif p < 0.01:
        label, y_text = '**', y + h * 2.5
    elif p < 0.05:
        label, y_text = '*', y + h * 2
    else:
        label, y_text = 'ns', y + h * 1.5
    ax.plot([x1, x1, x2, x2], [y, y + h * 0.5, y + h * 0.5, y], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y_text, label, ha='center', va='bottom', fontsize=10)


def plot_intra_inter_bar(stats_orig, stats_aug, save_dir):
    """Bar chart with significance brackets."""
    labels_bar = ['Intra-GAS', 'Intra-HPV_CA', 'Inter\n(GAS VS HPV_CA)']
    x = np.arange(len(labels_bar))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.rcParams.update({'font.size': 12})

    keys = ['intra_GAS', 'intra_HPV_CA', 'inter_GAS_HPVCA']
    means_orig = [np.mean(stats_orig[k]) for k in keys]
    stds_orig  = [np.std(stats_orig[k])  for k in keys]
    means_aug  = [np.mean(stats_aug[k])  for k in keys]
    stds_aug   = [np.std(stats_aug[k])   for k in keys]

    ax.bar(x - width/2, means_orig, width, label='Original',
           yerr=stds_orig, capsize=4, color='#E64B35', alpha=0.85)
    ax.bar(x + width/2, means_aug,  width, label='Augmented',
           yerr=stds_aug,  capsize=4, color='#3C5488',  alpha=0.85)


    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel('Mean Pearson Correlation')
    ax.set_title('Intra-class vs Inter-class Correlation:\nOriginal vs Augmented Training Data')
    ax.legend(fontsize=11)
    ax.set_ylim(0.0, 1.05)
    # ax.axhline(y=0, color='grey', linestyle='--', lw=0.8)
    apply_academic_axis(ax)
    plt.tight_layout()

    out_path = os.path.join(save_dir, 'intra_inter_bar_comparison.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close()


def plot_intra_inter_kde(stats_orig, stats_aug, save_dir):
    """KDE distribution plots for intra/inter correlations: Original vs Augmented."""
    labels_kde = ['Intra-GAS', 'Intra-HPV_CA', 'Inter (GAS VS HPV_CA)']
    keys = ['intra_GAS', 'intra_HPV_CA', 'inter_GAS_HPVCA']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    plt.rcParams.update({'font.size': 11})

    for ax, name, key in zip(axes, labels_kde, keys):
        sns.kdeplot(stats_orig[key], ax=ax, color='#E64B35', lw=2,
                    label='Original', fill=True, alpha=0.3)
        sns.kdeplot(stats_aug[key], ax=ax, color='#3C5488', lw=2,
                    label='Augmented', fill=True, alpha=0.3)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Density')
        ax.set_ylim(0.0, 8.5)
        ax.set_xlim(0.2, 1.05)
        ax.legend(fontsize=10)
        apply_academic_axis(ax)

    fig.suptitle('Intra-class vs Inter-class Correlation Distributions:\nOriginal vs Augmented', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'intra_inter_kde_comparison.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close()

def plot_DEG_KDE(matrix_train, aug_label0, aug_label1, y_train_series, gene_names, save_dir):
    """Plot KDE for selected DEGs comparing real vs generated data per class."""
    available_genes = [g for g in gene_names if g in matrix_train.columns]
    missing_genes = [g for g in gene_names if g not in matrix_train.columns]
    if missing_genes:
        print(f"Genes not found in data: {missing_genes}")
    if not available_genes:
        print("No requested genes found. Skipping DEG KDE plot.")
        return

    y_labels = y_train_series.values
    real_hpv = matrix_train.loc[y_labels == 'HPV_CA']
    real_gas = matrix_train.loc[y_labels == 'GAS']

    for gene in available_genes:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        plt.rcParams.update({'font.size': 11})

        data_pairs = [
            ('HPV_CA', real_hpv[gene].values, aug_label0[gene].values, '#E64B35'),
            ('GAS',    real_gas[gene].values, aug_label1[gene].values, '#3C5488'),
        ]

        for ax, (label, real_vals, gen_vals, color) in zip(axes, data_pairs):
            sns.kdeplot(real_vals, ax=ax, color=color, lw=2, label='Real', fill=True, alpha=0.3)
            sns.kdeplot(gen_vals,  ax=ax, color=color, lw=2, linestyle='--', label='Generated', fill=True, alpha=0.3)
            ax.set_title(f'{gene} – {label}', fontsize=13)
            ax.set_xlabel('Expression')
            ax.set_ylabel('Density')
            ax.legend(fontsize=10)
            apply_academic_axis(ax)

        fig.suptitle(f'{gene} Expression: Real vs Generated', fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'DEG_KDE_{gene}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()
        plt.close()

        # 保存 CSV: Real data — HPV_CA
        pd.DataFrame({'sample': real_hpv.index, 'value': real_hpv[gene].values}).to_csv(
            os.path.join(save_dir, f'DEG_KDE_{gene}_real_HPV_CA.csv'), index=False)
        # Real data — GAS
        pd.DataFrame({'sample': real_gas.index, 'value': real_gas[gene].values}).to_csv(
            os.path.join(save_dir, f'DEG_KDE_{gene}_real_GAS.csv'), index=False)
        # Generated data — HPV_CA
        pd.DataFrame({'sample': aug_label0.index, 'value': aug_label0[gene].values}).to_csv(
            os.path.join(save_dir, f'DEG_KDE_{gene}_generated_HPV_CA.csv'), index=False)
        # Generated data — GAS
        pd.DataFrame({'sample': aug_label1.index, 'value': aug_label1[gene].values}).to_csv(
            os.path.join(save_dir, f'DEG_KDE_{gene}_generated_GAS.csv'), index=False)
        print(f"Saved 4 CSV files for {gene}")


# Compute metrics
stats_orig = compute_intra_inter_metrics(sim_matrix_original, y_train_series)
stats_aug  = compute_intra_inter_metrics(sim_matrix_augmented,  y_aug_train_series)

# Summary + statistical tests + ratio info
df_summary = summarize_intra_inter(stats_orig, stats_aug)
# Bar chart with significance
plot_intra_inter_bar(stats_orig, stats_aug, SAVE_DIR)

# KDE distribution plot
plot_intra_inter_kde(stats_orig, stats_aug, SAVE_DIR)
# Delta visualization (Inter normalized to 1)
# DEG KDE plots (PGC and DNMT1)
plot_DEG_KDE(matrix_train, aug_label0, aug_label1, y_train_series,
             gene_names=['PGC', 'DNMT1'], save_dir=SAVE_DIR)


# ============================================================
# PART 2: Confusion Matrix Comparison
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Confusion Matrix Comparison")
print("=" * 60)

# Build test dataset (use the original test set split)
X_test = torch.tensor(matrix_test.values, dtype=torch.float)
test_dataset = build_hetero_graph_dataset(X_test, adj_PPI, adj_GRN, label_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

target_indices = [0, 1]

def get_all_predictions(model, loader, device):
    """Run inference and collect predictions."""
    model.eval()
    model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            logits = outputs['combined_out']
            preds = logits.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, target_indices, display_labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=target_indices)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(4.5, 3.5))
    plt.rcParams.update({'font.size': 14})
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=display_labels, yticklabels=display_labels,
                     annot_kws={"size": 16})
    ax.set_xticklabels(display_labels, rotation=0)
    ax.set_yticklabels(display_labels, rotation=90, va='center')
    plt.title(title, fontsize=18)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.tight_layout()
    full_path = os.path.join(SAVE_DIR, f"{filename}.pdf")
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {full_path}")
    plt.show()
    plt.close()
    return cm_normalized





print("\n" + "=" * 60)
print("PART 3: Ablation Study (Training without Augmentation)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
matrix = matrix_train
label_list = label_train

fold_idx = 5
train_index, val_index = list(skf.split(matrix, label_list))[fold_idx - 1]
print(f"\n========== Fold {fold_idx} ==========")

X_train_raw = matrix.iloc[train_index]
y_train_raw = label_list[train_index]
X_val_raw = matrix.iloc[val_index]
y_val_raw = label_list[val_index]

# No augmentation - use raw data only
X_train = torch.tensor(X_train_raw.values, dtype=torch.float)
y_train = y_train_raw
X_val = torch.tensor(X_val_raw.values, dtype=torch.float)
y_val = y_val_raw

print(f"Train size (No Augmentation): {X_train.shape[0]}")

train_dataset = build_hetero_graph_dataset(X_train, adj_PPI, adj_GRN, y_train)
val_dataset = build_hetero_graph_dataset(X_val, adj_PPI, adj_GRN, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


model = GraphLevelHeteroGCN(
    c_in=1,
    c_hidden=256,
    c_out=2,
    lr=5e-5,
    weight_decay=1e-3,
    dp_rate=0.01,
    dp_rate_linear=0.2,
    warmup_steps=50,
    label_smoothing=0.05
)
experiment_name = f"heterogcn_no_aug_fold_{fold_idx}"
trainer = create_trainer(
    max_epochs=500, patience=100, min_delta=1e-4,
    log_dir="lightning_logs_ablation",
    save_dir="checkpoints_ablation",
    experiment_name=experiment_name
)

trainer.fit(model, train_loader, val_loader)
print(f"Testing fold {fold_idx} (No Augmentation)...")


print("Ablation Study Completed.")
# --- Ablation model (no augmentation, fold 5) ---
# NOTE: This only works if you have run the ablation training loop below
# and saved checkpoints to checkpoints_ablation/

abl_ckpt_path = os.path.join(ABLATION_CKPT_DIR, f"heterogcn_no_aug_fold_5/epoch=497-val_total_loss=0.3714.ckpt")
best_abl_model = GraphLevelHeteroGCN.load_from_checkpoint(abl_ckpt_path)
best_model = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_5/epoch=412-val_total_loss=0.3654.ckpt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer.test(best_model, test_loader)
trainer.test(best_abl_model, test_loader)

# --- Best WEDGE model (with augmentation) ---
model_wedge = best_model
model_abl = best_abl_model
y_true_wedge, y_pred_wedge = get_all_predictions(model_wedge, test_loader, device)
plot_confusion_matrix(
    y_true_wedge, y_pred_wedge,
    target_indices=target_indices,
    display_labels=display_labels_list,
    title="WEDGE (with Augmentation)",
    filename="CM_WEDGE"
)

y_true_abl, y_pred_abl = get_all_predictions(model_abl, test_loader, device)
plot_confusion_matrix(
    y_true_abl, y_pred_abl,
    target_indices=target_indices,
    display_labels=display_labels_list,
    title="WEDGE (Ablation)",
    filename="CM_WEDGE_ABL"
)



