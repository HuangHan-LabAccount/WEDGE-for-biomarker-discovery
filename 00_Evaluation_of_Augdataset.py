import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.linalg import sqrtm
from sklearn.model_selection import StratifiedKFold
import os
import sys
import torch

sys.path.append('lib/')
from utilsdata import *
from WEDGE_model import *
from Train import *
from torch_geometric.data import DataLoader

# Get the directory where this script is located
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR
plt.rcParams['pdf.fonttype'] = 42

# ===================== Data & Model Paths =====================
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
STRING_DIR = os.path.join(PPI_DIR, 'String_database')
TRRUST_DIR = os.path.join(PPI_DIR, 'Trrust_database')

# New model checkpoints - 10 models per fold
models_paths = {
    1: [f"heterogcn_1/epoch={e}-val_total_loss={l}.ckpt" for e, l in [
        (674, 0.5289), (660, 0.5377), (605, 0.5420), (656, 0.5448), (609, 0.5426),
        (541, 0.5428), (476, 0.5443), (591, 0.5439), (340, 0.5475), (922, 0.5478)]],
    2: [f"heterogcn_2/epoch={e}-val_total_loss={l}.ckpt" for e, l in [
        (742, 0.4876), (857, 0.4892), (643, 0.4914), (329, 0.4870), (328, 0.4890),
        (307, 0.4872), (295, 0.4917), (274, 0.4909), (242, 0.4850), (211, 0.4868)]],
    3: [f"heterogcn_3/epoch={e}-val_total_loss={l}.ckpt" for e, l in [
        (608, 0.3583), (692, 0.3657), (560, 0.3677), (574, 0.3697), (578, 0.3672),
        (626, 0.3685), (760, 0.3692), (821, 0.3678), (828, 0.3696), (876, 0.3694)]],
    4: [f"heterogcn_4/epoch={e}-val_total_loss={l}.ckpt" for e, l in [
        (192, 0.3839), (207, 0.4058), (177, 0.4036), (174, 0.4247), (172, 0.4284),
        (162, 0.4158), (140, 0.4230), (123, 0.4125), (105, 0.4233), (58, 0.4077)]],
    5: [f"heterogcn_5/epoch={e}-val_total_loss={l}.ckpt" for e, l in [
        (411, 0.3632), (412, 0.3654), (413, 0.3663), (415, 0.3655), (416, 0.3656),
        (417, 0.3659), (505, 0.3689), (335, 0.3676), (337, 0.3692), (347, 0.3690)]],
}


# ===================== Part 1: FID Evaluation =====================

def calculate_fd(real_data, generated_data):
    """Calculate Fréchet Distance between real and generated data."""
    scaler = StandardScaler()
    real_data_scaled = scaler.fit_transform(real_data)
    generated_data_scaled = scaler.transform(generated_data)

    mu_real = np.mean(real_data_scaled, axis=0)
    mu_gen = np.mean(generated_data_scaled, axis=0)
    sigma_real = np.cov(real_data_scaled.T)
    sigma_gen = np.cov(generated_data_scaled.T)
    diff = mu_real - mu_gen

    try:
        sqrt_sigma = sqrtm(sigma_real @ sigma_gen)
        if np.iscomplexobj(sqrt_sigma):
            sqrt_sigma = sqrt_sigma.real
    except:
        sqrt_sigma = np.sqrt(np.diagonal(sigma_real) * np.diagonal(sigma_gen)).sum()

    fd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * sqrt_sigma)
    return fd


def generate_random_matrix_matched_stats(original_matrix, target_shape, random_state=42):
    np.random.seed(random_state)
    if isinstance(original_matrix, pd.DataFrame):
        original_values = original_matrix.values
    else:
        original_values = original_matrix

    means = np.mean(original_values, axis=0)
    stds = np.std(original_values, axis=0)
    n_samples, n_features = target_shape
    random_data = np.random.normal(0, 1, (n_samples, n_features))
    random_matrix = random_data * stds + means
    return random_matrix


def evaluate_gan_fd(matrix, aug_data_dir, num_folds=5):
    """Evaluate GAN fd scores and calculate one random baseline."""
    gan_results = {}
    random_baseline_fd = None

    for fold in range(1, num_folds + 1):
        try:
            aug_label0 = pd.read_csv(os.path.join(aug_data_dir, f'generated_data_fold{fold}_0.csv'))
            aug_label1 = pd.read_csv(os.path.join(aug_data_dir, f'generated_data_fold{fold}_1.csv'))

            if 'id' in aug_label0.columns:
                aug_label0 = aug_label0.drop(['id'], axis=1)
            if 'id' in aug_label1.columns:
                aug_label1 = aug_label1.drop(['id'], axis=1)

            combined_data = pd.concat([aug_label0, aug_label1], axis=0)
            common_features = matrix.columns.intersection(combined_data.columns)

            if len(common_features) == 0:
                print(f"Fold {fold}: No common features found!")
                continue

            real_data = matrix[common_features].values
            gen_data = combined_data[common_features].values
            gan_fd = calculate_fd(real_data, gen_data)
            gan_results[f'Fold_{fold}'] = gan_fd

            if fold == 1:
                target_shape = gen_data.shape
                random_matrix = generate_random_matrix_matched_stats(
                    matrix[common_features], target_shape=target_shape, random_state=42)
                random_baseline_fd = calculate_fd(real_data, random_matrix)
                print(f"Random baseline fd: {random_baseline_fd:.4f}")

            print(f"Fold {fold} - GAN fd: {gan_fd:.4f}")

        except Exception as e:
            print(f"Error processing fold {fold}: {e}")

    return gan_results, random_baseline_fd


def plot_fd_results(gan_results, random_baseline_fd, save_path=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)
    methods = list(gan_results.keys()) + ['Random']
    values = list(gan_results.values()) + [random_baseline_fd]
    n_methods = len(methods)
    colors = ["#E64B35", "#3C5488", "#00A089", "#F5AE65", "#4DBBD7", "#785391", "#F29B81", "#8DD3C7"]
    pos = np.arange(n_methods)
    bar_width = 0.6

    bars = plt.bar(pos, values, bar_width, alpha=0.8,
                   color=[colors[i % len(colors)] for i in range(n_methods)])

    for bar, score in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(pos, methods)
    plt.ylabel('FD Score', fontsize=10)
    plt.title('FD Scores: GAN Performance vs Random Baseline', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    return plt.gcf()


def save_fd_results(gan_results, random_baseline_fd, output_dir):
    comparison_data = []
    for fold in gan_results.keys():
        comparison_data.append({'Method': fold, 'FD': gan_results[fold], 'Type': 'GAN'})
    comparison_data.append({'Method': 'Random', 'FD': random_baseline_fd, 'Type': 'Baseline'})
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'fd_results.csv'), index=False)
    print(f"fd results saved to {output_dir}/fd_results.csv")


# ===================== Part 2: Accuracy Evaluation =====================

def evaluate_model_with_trainer(trainer, model, test_loader):
    """Use trainer to evaluate model, return three accuracy metrics on test set."""
    test_results = trainer.test(model, test_loader)
    return {
        'test_protein_acc': test_results[0]['test_protein_acc'],
        'test_gene_acc': test_results[0]['test_gene_acc'],
        'test_combined_acc': test_results[0]['test_combined_acc']
    }


def evaluate_all_models(matrix_train, label_train, matrix_test, label_test,
                       adj_PPI, adj_GRN, models_paths,
                       aug_data_dir, checkpoint_dir, num_folds=5):
    """Evaluate all folds and all models' accuracy."""
    results = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold in range(1, num_folds + 1):
        print(f"\n{'='*60}\nProcessing Fold {fold}\n{'='*60}")

        # Load augmented data
        aug_label0 = pd.read_csv(os.path.join(aug_data_dir, f'generated_data_fold{fold}_0.csv'))
        aug_label1 = pd.read_csv(os.path.join(aug_data_dir, f'generated_data_fold{fold}_1.csv'))
        aug_label0.index = aug_label0['id']
        aug_label1.index = aug_label1['id']

        # Split data
        fold_results = list(skf.split(matrix_train, label_train))
        train_index, val_index = fold_results[fold - 1]

        y_train = label_train[train_index]
        y_train = torch.tensor(np.concatenate([y_train,
                                                aug_label0['label'].values,
                                                aug_label1['label'].values]))
        y_val = label_train[val_index]

        aug_label0 = aug_label0.drop(columns=['subset', 'label', 'id'])
        aug_label1 = aug_label1.drop(columns=['subset', 'label', 'id'])
        X_train = matrix_train.iloc[train_index]
        X_train = pd.concat([X_train, aug_label0, aug_label1], axis=0)
        X_val = matrix_train.iloc[val_index]

        X_train = torch.tensor(X_train.values, dtype=torch.float)
        X_val = torch.tensor(X_val.values, dtype=torch.float)
        X_test = torch.tensor(matrix_test.values, dtype=torch.float)
        y_test = label_test

        # Build graph datasets
        train_dataset = build_hetero_graph_dataset(X_train, adj_PPI, adj_GRN, y_train)
        val_dataset = build_hetero_graph_dataset(X_val, adj_PPI, adj_GRN, y_val)
        test_dataset = build_hetero_graph_dataset(X_test, adj_PPI, adj_GRN, y_test)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Evaluate 10 models for this fold
        fold_models = models_paths[fold]
        for model_idx, model_name in enumerate(fold_models, 1):
            model_path = os.path.join(checkpoint_dir, model_name)
            print(f"  Model {model_idx}: {model_name}")

            try:
                model = GraphLevelHeteroGCN.load_from_checkpoint(model_path)
                metrics = evaluate_model_with_trainer(trainer, model, test_loader)
                results.append({
                    'Fold': fold,
                    'Model': model_idx,
                    'Model_Path': model_name,
                    'Test_Protein_Acc': metrics['test_protein_acc'],
                    'Test_Gene_Acc': metrics['test_gene_acc'],
                    'Test_Combined_Acc': metrics['test_combined_acc']
                })
                print(f"    Protein: {metrics['test_protein_acc']:.4f}, "
                      f"Gene: {metrics['test_gene_acc']:.4f}, "
                      f"Combined: {metrics['test_combined_acc']:.4f}")
            except Exception as e:
                print(f"    Failed: {e}")
                results.append({
                    'Fold': fold,
                    'Model': model_idx,
                    'Model_Path': model_name,
                    'Test_Protein_Acc': None,
                    'Test_Gene_Acc': None,
                    'Test_Combined_Acc': None
                })

    return pd.DataFrame(results)


def plot_accuracy_results(results_df, save_path=None, figsize=(6.5, 4)):
    """Plot bar chart of three accuracy metrics per fold."""
    plt.figure(figsize=figsize)
    bar_width = 0.25
    metrics = ['Test_Protein_Acc', 'Test_Gene_Acc', 'Test_Combined_Acc']
    metric_labels = ['PPI', 'GRN', 'Combined']
    folds = sorted(results_df['Fold'].unique())
    colors = ["#E64B35", "#3C5488", "#00A089"]

    x = np.arange(len(folds))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values, errors = [], []
        for fold in folds:
            fold_data = results_df[results_df['Fold'] == fold]
            values.append(fold_data[metric].mean())
            errors.append(fold_data[metric].std())

        pos = x + bar_width * (i - 1)
        bars = plt.bar(pos, values, bar_width, alpha=0.8, color=color, label=label,
                       capsize=3, ecolor='black', error_kw={'elinewidth': 1})

        for bar, val, err in zip(bars, values, errors):
            plt.text(bar.get_x() + bar.get_width() / 2, val + err + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(x, [f'Fold {fold}' for fold in folds], fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Accuracy Comparison Across Different Folds', fontsize=14)
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy figure saved to: {save_path}")
    else:
        plt.show()
    return plt.gcf()


def save_accuracy_results(results_df, output_dir):
    results_df.to_csv(os.path.join(output_dir, 'accuracy_results.csv'), index=False)
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    for fold in sorted(results_df['Fold'].unique()):
        fold_data = results_df[results_df['Fold'] == fold]
        print(f"\nFold {fold}:")
        for metric, label in [('Test_Protein_Acc', 'PPI'),
                               ('Test_Gene_Acc', 'GRN'),
                               ('Test_Combined_Acc', 'Combined')]:
            vals = fold_data[metric].dropna()
            if len(vals) > 0:
                print(f"  {label} - Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
    print(f"\nAccuracy results saved to {output_dir}/accuracy_results.csv")


# ===================== Main Execution =====================

if __name__ == '__main__':
    metric_dir = os.path.join(OUTPUT_DIR, 'metric', 'GAN_metric')
    os.makedirs(metric_dir, exist_ok=True)

    # ---- Part 1: fd Evaluation ----
    print("\n" + "=" * 60)
    print("PART 1: fd EVALUATION")
    print("=" * 60)

    matrix_demo = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T
    print(f"Demo data shape: {matrix_demo.shape}")

    gan_results, random_baseline_fd = evaluate_gan_fd(matrix_demo, AUG_DATA_DIR)
    save_fd_results(gan_results, random_baseline_fd, metric_dir)

    fd_df = pd.read_csv(os.path.join(metric_dir, 'fd_results.csv'))
    random_baseline_fd = fd_df.loc[fd_df['Type'] == 'Baseline', 'FD'].values[0]
    gan_dict = dict(zip(fd_df.loc[fd_df['Type'] == 'GAN', 'Method'],
                        fd_df.loc[fd_df['Type'] == 'GAN', 'FD']))
    plot_fd_results(gan_dict, random_baseline_fd,
                     save_path=os.path.join(metric_dir, 'fd_comparison.pdf'))

    # ---- Part 2: Accuracy Evaluation ----
    print("\n" + "=" * 60)
    print("PART 2: ACCURACY EVALUATION")
    print("=" * 60)

    # Load data from demo data directory
    meta = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_selected.csv'), index_col=0)
    matrix = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T
    matrix_Degene_sub = DEgene_selected(matrix, path=PROJECT_ROOT)
    matrix = matrix_Degene_sub
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    encoder = LabelEncoder()
    matrix_all = matrix_sub
    label_all = (torch.tensor(encoder.fit_transform(meta_sub.CancerType.values), dtype=torch.float))
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    protein_matrix_PPI = load_Stringdatabase(path=STRING_DIR,
                                             file_name="human_PPI_score_Stringdatabase(700up).csv")
    protein_matrix_GRN = load_Stringdatabase(path=TRRUST_DIR,
                                             file_name="TF_filtered_human.csv")
    adj_PPI = getAdjByString(protein_matrix_PPI, matrix_train, one_direction=False)
    adj_GRN = getAdjByString(protein_matrix_GRN, matrix_train, one_direction=True)
    gene_features = torch.tensor(matrix_train.values, dtype=torch.float)
    encoder = LabelEncoder()
    encoder.fit(meta_train.CancerType.values)  # Fit encoder on training labels
    label_test = torch.tensor(encoder.transform(meta_test.CancerType.values), dtype=torch.float)

    # Create trainer
    trainer = create_trainer(max_epochs=150, patience=1000, min_delta=1e-4,
                            log_dir="lightning_logs", save_dir="checkpoints",
                            experiment_name="heterogcn_fold")
    # Evaluate all models
    results_df = evaluate_all_models(
        matrix_train, label_train, matrix_test, label_test,
        adj_PPI, adj_GRN,
        models_paths = models_paths,
        aug_data_dir=AUG_DATA_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        num_folds=5
    )

    save_accuracy_results(results_df, metric_dir)

    # Plot accuracy
    plot_accuracy_results(
        results_df,
        save_path=os.path.join(metric_dir, 'accuracy_comparison.pdf')
    )
