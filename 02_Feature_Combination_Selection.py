# tensorboard --logdir H:/Proteomic/lighting_loggers/GraphLevelProteinGCN/
# tensorboard --logdir H:/Proteomic/lighting_loggers/GraphLevelProteinGCN2/
# http://localhost:6006/

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import os
import warnings
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
import igraph as ig
sys.path.append('lib/')
from utilsdata import *
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42  # 保持矢量字体可编辑性
})


# Get the directory where this script is located
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR

# Data paths - using relative paths from project root
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
STRING_DIR = os.path.join(PPI_DIR, 'String_database')
TRRUST_DIR = os.path.join(PPI_DIR, 'Trrust_database')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
GENE_RANK_DIR = os.path.join(OUTPUT_DIR, 'gene_rank')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
warnings.filterwarnings("ignore")

if True:
    def SplitData(matrix, meta):
        meta_train = meta[meta['Batch'] == 1]
        meta_test = meta[meta['Batch'] == 2]
        matrix_train = matrix.loc[meta_train['MS_number']]
        matrix_test = matrix.loc[meta_test['MS_number']]
        return matrix_train, matrix_test, meta_train, meta_test


    def generate_random_gene_combinations(deg_data, n_combinations=10000, genes_per_combo=3, if_all=False):
        num_genes = len(deg_data)
        rd_set = set()

        if if_all:
            all_combinations = list(itertools.combinations(deg_data.iloc[:, 0], genes_per_combo))
            return all_combinations

        else:
            while len(rd_set) < n_combinations:
                indices = random.sample(range(num_genes), genes_per_combo)
                genes = tuple(sorted(str(deg_data.iloc[i, 0]) for i in indices))

                if not any(gene in ('nan', 'Unknown', 'None') for gene in genes):
                    rd_set.add(genes)
            return list(rd_set)
        return list(rd_set)


    def train_fold(X_train, y_train, X_val, y_val, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        # X_train_scaled = X_train
        # X_val_scaled = X_val
        # X_test_scaled = X_test
        # 模型训练
        clf = LogisticRegressionCV(
            Cs=[25],
            penalty='elasticnet',
            fit_intercept=False,
            cv=5,
            solver='saga',
            n_jobs=1,
            refit=True,
            class_weight='balanced',
            multi_class='ovr',
            max_iter=1000,
            l1_ratios=[0.5]
        )
        clf.fit(X_train_scaled, y_train)

        # 获取系数
        coef = clf.coef_.ravel()
        result = [np.concatenate([clf.intercept_, coef])]

        # 预测概率和预测标签
        y_val_scores = clf.predict_proba(X_val_scaled)
        y_test_scores = clf.predict_proba(X_test_scaled)

        # 预测标签（用于准确率计算）
        y_val_pred = clf.predict(X_val_scaled)
        y_test_pred = clf.predict(X_test_scaled)

        # 计算每个类的AUC（如果是二分类，只取正类的概率）
        if y_val_scores.shape[1] == 2:
            # 二分类情况：只取正类（类别1）的概率
            y_val_scores_binary = y_val_scores[:, 1]
            y_test_scores_binary = y_test_scores[:, 1]

            try:
                val_auc = roc_auc_score(y_val, y_val_scores_binary)
                test_auc = roc_auc_score(y_test, y_test_scores_binary)
                val_aucs = [val_auc]
                test_aucs = [test_auc]
            except ValueError as e:
                print(f"AUC calculation error: {e}")
                val_aucs = [np.nan]
                test_aucs = [np.nan]
        else:
            onehot_encoder = OneHotEncoder(sparse_output=False)
            y_val_onehot = onehot_encoder.fit_transform(y_val.reshape(-1, 1))
            y_test_onehot = onehot_encoder.transform(y_test.reshape(-1, 1))

            val_aucs = []
            test_aucs = []

            for i in range(y_val_onehot.shape[1]):
                try:
                    auc_val = roc_auc_score(y_val_onehot[:, i], y_val_scores[:, i])
                    val_aucs.append(auc_val)

                    auc_test = roc_auc_score(y_test_onehot[:, i], y_test_scores[:, i])
                    test_aucs.append(auc_test)
                except ValueError as e:
                    print(f"AUC calculation error for class {i}: {e}")
                    val_aucs.append(np.nan)
                    test_aucs.append(np.nan)

        # 计算准确率
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        return y_val, y_val_pred, y_test, y_test_pred, val_aucs, test_aucs, val_acc, test_acc, result


    def evaluate_gene_combinations(matrix, label, gene_combinations, matrix_test, label_test,
                                   n_splits=5, n_repeats=3):
        results = []
        y = label
        y_test = label_test

        for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations),
                                     desc="Gene combinations"):
            X = matrix.loc[:, genes]
            X_test = matrix_test.loc[:, genes].values
            fold_val_aucs = []  # 验证集AUC
            fold_test_aucs = []  # 测试集AUC
            fold_val_accs = []  # 验证集准确率
            fold_test_accs = []  # 测试集准确率

            for repeat in range(n_repeats):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_train = X.iloc[train_idx].values  # 转换为numpy数组
                    X_val = X.iloc[val_idx].values

                    y_train, y_val = y[train_idx], y[val_idx]

                    # 注意：这里接收更新后的返回值
                    y_val_actual, y_val_pred, y_test_actual, y_test_pred, val_aucs, test_aucs, val_acc, test_acc, _ = train_fold(
                        X_train, y_train, X_val, y_val, X_test, y_test)

                    fold_val_aucs.append(val_aucs)
                    fold_test_aucs.append(test_aucs)
                    fold_val_accs.append(val_acc)
                    fold_test_accs.append(test_acc)

            # 转换为numpy数组便于计算
            fold_val_aucs = np.array(fold_val_aucs)
            fold_test_aucs = np.array(fold_test_aucs)
            fold_val_accs = np.array(fold_val_accs)
            fold_test_accs = np.array(fold_test_accs)

            # 计算平均值和标准差
            mean_val_aucs = np.mean(fold_val_aucs, axis=0)
            std_val_aucs = np.std(fold_val_aucs, axis=0)
            mean_test_aucs = np.mean(fold_test_aucs, axis=0)
            std_test_aucs = np.std(fold_test_aucs, axis=0)

            # 计算准确率的平均值和标准差
            mean_val_accs = np.mean(fold_val_accs)
            std_val_accs = np.std(fold_val_accs)
            mean_test_accs = np.mean(fold_test_accs)
            std_test_accs = np.std(fold_test_accs)

            # 存储结果
            results.append({
                'genes': genes,
                'mean_val_aucs': mean_val_aucs,
                'std_val_aucs': std_val_aucs,
                'mean_test_aucs': mean_test_aucs,
                'std_test_aucs': std_test_aucs,
                'mean_val_accs': mean_val_accs,
                'std_val_accs': std_val_accs,
                'mean_test_accs': mean_test_accs,
                'std_test_accs': std_test_accs,
            })

            print(f"Combination {combo_idx + 1}/{len(gene_combinations)}")
            print(f"Validation AUC: {mean_val_aucs} ± {std_val_aucs}")
            print(f"Test AUC: {mean_test_aucs} ± {std_test_aucs}")
            print(f"Validation Accuracy: {mean_val_accs:.4f} ± {std_val_accs:.4f}")
            print(f"Test Accuracy: {mean_test_accs:.4f} ± {std_test_accs:.4f}")

        return pd.DataFrame(results)


    def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None, filename=None):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.close('all')  # 关闭所有之前的图形
        plt.figure(figsize=(4.5, 3.5))

        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})

        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')

        # 设置坐标轴范围
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # 设置标题和轴标签
        plt.title(title, fontsize=18)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)

        # 设置图例
        plt.legend(loc="lower right", fontsize=14)

        # 设置刻度标签字体大小
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # 关闭网格
        plt.grid(False)

        plt.tight_layout()

        if save_path:
            if filename:
                plt.savefig(f"{save_path}/{filename}_plot_roc.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{save_path}/plot_roc.pdf", bbox_inches='tight')

        plt.show()
        return roc_auc


    def plot_confusion_matrix(y_true, y_pred, labels=['HPV_CA', 'GAS'], title="Confusion Matrix", save_path=None,
                              filename=None):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.close('all')  # 关闭所有之前的图形
        plt.figure(figsize=(4.5, 3.5))
        plt.rcParams.update({'font.size': 14})
        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.title(title, fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        if save_path:
            if filename:
                plt.savefig(f"{save_path}/{filename}_confusion_matrix.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{save_path}/confusion_matrix.pdf", bbox_inches='tight')

        plt.show()


    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model", title='title', save_path=None,
                       filename=None,
                       **kwargs):
        # 预测概率（用于ROC曲线）
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # 获取预测标签（用于混淆矩阵和准确率）
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算准确率
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # 计算并绘制ROC曲线
        print(f"\n{model_name} - ROC Curves:")
        train_auc = plot_roc_curve(y_train, y_train_prob, title=f"Training ROC Curve",
                                   save_path=save_path, filename=f"{filename}_train_roc" if filename else None)
        test_auc = plot_roc_curve(y_test, y_test_prob, title=f"Testing ROC Curve",
                                  save_path=save_path, filename=f"{filename}_test_roc" if filename else None)

        # 计算并绘制混淆矩阵
        print(f"\n{model_name} - Confusion Matrices:")
        train_cm = plot_confusion_matrix(y_train, y_train_pred,
                                         title=f"{title}",
                                         save_path=save_path, filename=f"{filename}_train_cm" if filename else None)
        test_cm = plot_confusion_matrix(y_test, y_test_pred,
                                        title=f"{title}",
                                        save_path=save_path, filename=f"{filename}_test_cm" if filename else None)

        # 打印评估结果
        print(f"\n{model_name} - Performance Metrics:")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Testing AUC: {test_auc:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")

        # 返回评估指标
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_confusion_matrix': train_cm,
            'test_confusion_matrix': test_cm
        }
if True:
    # Load data from Demo_data
    meta = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_selected.csv'), index_col=0)
    # Sample * Protein
    matrix = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T
    # For DE gene selection, use a default list or load from PPI_DIR if available
    # HPV_EA_DEG = pd.read_csv(f'{path}/data/DEG/HPV_EA_DEG_1.5.csv', index_col=0)
    HPV_EA_DEG = pd.read_csv(os.path.join(DEMO_DATA_DIR,"DEG", 'HPV_EA_DEG_1.5.csv'), index_col=0)
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)

    X_train = matrix_train
    X_test = matrix_test

    y_train = meta_train.CancerType
    y_test = meta_test.CancerType
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
import json
from joblib import dump, load

RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
# Based on Interpret PPI/GRN top 20 + top 20 nodes
HPV_EA_DEG = pd.read_csv(f'{GENE_RANK_DIR}/union_gene.csv', index_col=0)
os.makedirs(RESULTS_DIR, exist_ok=True)
n_genes = 3
combinations_file = os.path.join(RESULTS_DIR, f'gene_combinations_{n_genes}genes.json')
gene_counts = [3, 4, 5, 6, 7, 8, 9, 10]
for n_genes in gene_counts:
    print(f"\n{'=' * 60}")
    print(f"Processing {n_genes}-gene combinations")
    print(f"{'=' * 60}")

    combinations_file = os.path.join(RESULTS_DIR, f'gene_combinations_{n_genes}genes.json')

    if os.path.exists(combinations_file):
        print(f"Found existing gene combinations file: {combinations_file}")
        print(f"Skipping generation for {n_genes}-gene combinations...")
    else:
        print(f"Generating {n_genes}-gene combinations...")

        if n_genes in [1, 2]:
            gene_combinations = generate_random_gene_combinations(
                HPV_EA_DEG,
                n_combinations=4000,
                genes_per_combo=n_genes,
                if_all=True
            )
        else:
            gene_combinations = generate_random_gene_combinations(
                HPV_EA_DEG,
                n_combinations=4000,
                genes_per_combo=n_genes
            )

        with open(combinations_file, 'w') as f:
            json.dump(gene_combinations, f)
        print(f"Saved gene combinations to: {combinations_file}")

    with open(combinations_file, 'r') as f:
        loaded_gene_combinations = json.load(f)
    print(f"Loaded {len(loaded_gene_combinations)} gene combinations")

    joblib_file = os.path.join(RESULTS_DIR, f'results_df_{n_genes}genes.joblib')
    csv_file = os.path.join(RESULTS_DIR, f'4000_combos_{n_genes}genes.csv')

    if os.path.exists(joblib_file):
        print(f"Found existing evaluation results: {joblib_file}")
        print(f"Skipping evaluation for {n_genes}-gene combinations...")
        results_df = load(joblib_file)
    else:
        print(f"Evaluating {n_genes}-gene combinations...")
        results_df = evaluate_gene_combinations(
            matrix=X_train,
            gene_combinations=loaded_gene_combinations,
            label=y_train,
            matrix_test=X_test,
            label_test=y_test,
            n_splits=5,
            n_repeats=3
        )

        results_df['combined_score_AUC'] = (results_df['mean_val_aucs'] * 1).astype(float)
        dump(results_df, joblib_file)
        print(f"Saved results (joblib) to: {joblib_file}")
        results_df.to_csv(csv_file, index=False)
        print(f"Saved results (CSV) to: {csv_file}")


gene_counts = [3, 4, 5, 6, 7, 8, 9, 10]
for n_genes in gene_counts:
    print(f"\n{'=' * 60}")
    print(f"Processing {n_genes}-gene combinations")
    print(f"{'=' * 60}")

    combinations_file = os.path.join(RESULTS_DIR, f'gene_combinations_{n_genes}genes.json')

    if os.path.exists(combinations_file):
        print(f"Found existing gene combinations file: {combinations_file}")
        print(f"Skipping generation for {n_genes}-gene combinations...")
    with open(combinations_file, 'r') as f:
        loaded_gene_combinations = json.load(f)
    print(f"Loaded {len(loaded_gene_combinations)} gene combinations")

    joblib_file = os.path.join(RESULTS_DIR, f'results_df_{n_genes}genes.joblib')
    csv_file = os.path.join(RESULTS_DIR, f'4000_combos_{n_genes}genes.csv')

    if os.path.exists(joblib_file):
        print(f"Found existing evaluation results: {joblib_file}")
        print(f"Skipping evaluation for {n_genes}-gene combinations...")
        results_df = load(joblib_file)
        results_df['combined_score_AUC'] = (results_df['mean_val_aucs']).astype(float)
        dump(results_df, joblib_file)
        print(f"Saved results (joblib) to: {joblib_file}")
        results_df['Score'] = results_df['combined_score_AUC']
        results_df = results_df.drop(columns=['combined_score_AUC'])
        results_df.to_csv(csv_file, index=False)
        print(f"Saved results (CSV) to: {csv_file}")
# plot for genes 3 - 10
if True:
    # Define the counts you want to loop through
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import pandas as pd
    import numpy as np
    from collections import Counter
    from joblib import load
    import os

    # Define the counts you want to loop through
    gene_counts_list = [3, 4, 5, 6, 7, 8, 9, 10]

    # Ensure the output directory exists
    save_base_path = os.path.join(OUTPUT_DIR, 'fig/Generank')
    os.makedirs(save_base_path, exist_ok=True)

    for n_genes in gene_counts_list:
        print(f"\nProcessing Feature Frequency for {n_genes} genes...")

        # 1. Load the specific results file
        file_path = os.path.join(RESULTS_DIR, f'results_df_{n_genes}genes.joblib')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue

        results_df = load(file_path)

        # 2. Identify the correct scoring column
        if 'combined_score_AUC' in results_df.columns:
            sort_col = 'combined_score_AUC'
        elif 'combined_score' in results_df.columns:
            sort_col = 'combined_score'
        else:
            print(f"Score column not found in {file_path}. Skipping.")
            continue

        # 3. Get Top 40 Combinations based on the score
        top50_index = results_df[sort_col].sort_values(ascending=False).index[:50]
        top50_combos = results_df.loc[top50_index].genes

        # 4. Count Gene Frequencies
        all_genes = [gene for combo in top50_combos for gene in combo]
        gene_counts_dict = Counter(all_genes)

        genes = list(gene_counts_dict.keys())
        counts = list(gene_counts_dict.values())

        df_plot = pd.DataFrame({'Gene': genes, 'Frequency': counts})
        df_plot = df_plot.sort_values('Frequency', ascending=False).reset_index(drop=True)

        # Optional: If there are too many genes (e.g. >30), limit the plot to the top 20-25
        # to keep it readable like the reference image.
        if len(df_plot) > 20:
            df_plot = df_plot.head(20)

        # 5. Plotting
        plt.close('all')
        # Increase width slightly to accommodate labels if many genes exist
        plt.figure(figsize=(8, 4.5))

        # --- COLOR GENERATION ---
        # Create a gradient based on the Frequency values.
        # We use a Normalize object to map frequency values to [0,1]
        norm = plt.Normalize(df_plot['Frequency'].min(), df_plot['Frequency'].max())
        # Use the 'Greens' colormap
        # We maintain a minimum intensity (0.3) so the lowest bars aren't invisible white
        map_values = 0.3 + 0.7 * (df_plot['Frequency'] - df_plot['Frequency'].min()) / (
                df_plot['Frequency'].max() - df_plot['Frequency'].min())
        colors = cm.Greens(map_values)

        # Draw Bars
        bars = plt.bar(df_plot['Gene'], df_plot['Frequency'], color=colors, width=0.7)

        # --- ADD VALUE LABELS ---
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=10, color='black', fontweight='regular')

        # Styling
        plt.title(f'Feature Frequency in Top-Performing Combinations ({n_genes} Genes)', fontsize=16)
        plt.ylabel('Recurrence Frequency', fontsize=14)
        # plt.xlabel('Genes', fontsize=14) # X-label often omitted if gene names are self-explanatory

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        # Remove top and right spines for a cleaner look (like the reference)
        ax = plt.gca()
        ax.spines['top'].set_visible(True)  # The reference has a box, so keep True. Set False if you want open look.
        ax.spines['right'].set_visible(True)

        # Axis formatting
        if not df_plot.empty:
            y_max = max(df_plot['Frequency']) * 1.15  # Add 15% headroom for the text labels
            ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
            plt.ylim(0, y_max)
            plt.xlim(-0.7, len(df_plot) - 0.3)

        plt.tight_layout()

        # 6. Save Plot

        save_name = f'gene_frequency_distribution_{n_genes}genes_top50_styled.pdf'
        full_save_path = os.path.join(save_base_path, save_name)

        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# ============================================================

def plot_top10_gene_frequency_line(gene_counts_list, results_dir, output_dir, top_n=30):
    """
    For each n_genes (3–10), load top-50 combos, count gene frequencies,
    then plot a line chart of the overall top-10 most frequent genes
    across all n_genes values.

    Parameters
    ----------
    gene_counts_list : list of int
        Gene counts to process, e.g. [3,4,5,6,7,8,9,10]
    results_dir : str
        Path to RESULTS_DIR containing results_df_{n_genes}genes.joblib
    output_dir : str
        Base output directory for figures
    top_n : int
        Number of top genes to keep in the final plot
    """
    from collections import Counter

    # Step 1: Aggregate gene frequencies across ALL n_genes
    global_counter = Counter()
    per_n_counter = {}   # n_genes -> Counter of gene frequencies

    for n_genes in gene_counts_list:
        file_path = os.path.join(results_dir, f'results_df_{n_genes}genes.joblib')
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        results_df = load(file_path)

        # Get top-50 combos by score
        sort_col = 'combined_score_AUC' if 'combined_score_AUC' in results_df.columns else \
                   'combined_score' if 'combined_score' in results_df.columns else \
                   results_df.columns[0]
        top_idx = results_df[sort_col].sort_values(ascending=False).index[:50]
        top_combos = results_df.loc[top_idx, 'genes']

        # Count gene frequencies for this n_genes
        counter = Counter(g for combo in top_combos for g in combo)
        per_n_counter[n_genes] = counter
        global_counter.update(counter)

    if not per_n_counter:
        print("No results loaded. Aborting.")
        return

    # Step 2: Identify top-N genes by total frequency across all n_genes
    top_genes = [g for g, _ in global_counter.most_common(top_n)]
    print(f"Top {top_n} genes: {top_genes}")

    # Step 3: Build DataFrame for line plot
    # rows = gene, columns = n_genes values, values = frequency
    freq_matrix = pd.DataFrame(index=top_genes, columns=sorted(per_n_counter.keys()), dtype=float)
    for n_genes in freq_matrix.columns:
        counter = per_n_counter[n_genes]
        for gene in top_genes:
            freq_matrix.loc[gene, n_genes] = counter.get(gene, 0)

    freq_matrix = freq_matrix.astype(float)

    # Step 4: Plot heatmap (R pheatmap style: blue-white-red, no clustering)
    save_base_path = os.path.join(output_dir, 'fig/Generank')
    os.makedirs(save_base_path, exist_ok=True)

    # Blue-white-red diverging colormap (matches R pheatmap style)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_white_red',
        ['#2E86AB', 'white', '#A23B72']
    )

    # Figure: rows=genes, cols=n_genes, so wider is better
    n_rows = len(top_genes)
    n_cols = len(freq_matrix.columns)
    fig_w = max(6, n_cols * 1.0 + 2)
    fig_h = max(6, n_rows * 0.3 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.rcParams.update({'font.size': 18})

    sns.heatmap(freq_matrix, ax=ax,
                cmap=cmap,
                # annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Recurrence Frequency'},
                linewidths=0.3,
                linecolor='white',
                annot_kws={'size': 11},
                xticklabels=True,
                yticklabels=True,
                square=False)

    ax.set_xlabel('Number of Features per Combination', fontsize=18)
    ax.set_ylabel('Protein', fontsize=18)
    ax.set_title(f'Top {top_n} Features Frequency Across Combination Sizes (3–10)', fontsize=18)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    out_path = os.path.join(save_base_path, f'top{top_n}_features_frequency_heatmap.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    plt.close()

    # Also save the frequency table
    csv_path = os.path.join(save_base_path, f'top{top_n}_features_frequency_table.csv')
    freq_matrix.to_csv(csv_path)
    print(f"Saved table: {csv_path}")

    return freq_matrix
gene_counts_list = [3, 4, 5, 6, 7, 8, 9, 10]
freq_table = plot_top10_gene_frequency_line(
    gene_counts_list=gene_counts_list,
    results_dir=RESULTS_DIR,
    output_dir=OUTPUT_DIR,
    top_n=30
)



# ==============================================================================
# STEP 1 — Compute and save all node metrics (run once)
# ==============================================================================

# ============================================================
# ============================================================
if True:
    def compute_and_save_network_metrics(ppi_path, grn_path, output_dir, ppi_score_thresh=700):
        """
        全程使用 igraph 计算所有网络指标，同时保存 raw（原始）和 normalized（归一化）版本。
        共 8 个文件（4 raw + 4 normalized）。

        Files saved (8 total):
            PPI  (undirected):
                ppi_degree-raw.csv / ppi_degree-norm.csv
                ppi_betweenness-raw.csv / ppi_betweenness-norm.csv
            GRN  (directed):
                grn_in_degree-raw.csv / grn_in_degree-norm.csv
                grn_out_degree-raw.csv / grn_out_degree-norm.csv
                grn_betweenness-raw.csv / grn_betweenness-norm.csv

        Normalization:
            Degree     -> / (n-1)
            Betweenness -> / ((n-1)(n-2)/2)  [undirected]  or / ((n-1)(n-2))  [directed]
        """
        save_dir = os.path.join(output_dir, 'network_metrics')
        os.makedirs(save_dir, exist_ok=True)

        def _save(name, df):
            path = os.path.join(save_dir, name)
            if os.path.exists(path):
                print(f'  {name} exists — skipping')
                return
            df.to_csv(path, index=False)
            print(f'  Saved: {name}')

        # ── PPI (undirected) ──────────────────────────────────────────────────────
        print('Building PPI network (igraph)...')
        ppi_df = pd.read_csv(ppi_path)
        ppi_df = ppi_df[ppi_df['combined_score'] >= ppi_score_thresh]
        g_ppi = ig.Graph.TupleList(
            ppi_df[['protein1', 'protein2']].itertuples(index=False),
            directed=False
        )
        n_ppi = g_ppi.vcount()
        print(f'  PPI: nodes={n_ppi}, edges={g_ppi.ecount()}')

        deg_raw = g_ppi.degree()
        deg_norm = [d / (n_ppi - 1) for d in deg_raw]
        _save('ppi_degree-raw.csv', pd.DataFrame({'protein': g_ppi.vs['name'], 'degree': deg_raw}))
        _save('ppi_degree-norm.csv', pd.DataFrame({'protein': g_ppi.vs['name'], 'degree': deg_norm}))

        bet_raw = g_ppi.betweenness(directed=False, normalized=False)
        max_bet_ppi = (n_ppi - 1) * (n_ppi - 2) / 2
        bet_norm = [b / max_bet_ppi for b in bet_raw]
        _save('ppi_betweenness-raw.csv', pd.DataFrame({'protein': g_ppi.vs['name'], 'betweenness': bet_raw}))
        _save('ppi_betweenness-norm.csv', pd.DataFrame({'protein': g_ppi.vs['name'], 'betweenness': bet_norm}))

        # ── GRN (directed) ─────────────────────────────────────────────────────────
        print('Building GRN network (igraph)...')
        grn_df = pd.read_csv(grn_path)
        g_grn = ig.Graph.TupleList(
            grn_df[['TF', 'Target']].itertuples(index=False),
            directed=True
        )
        n_grn = g_grn.vcount()
        print(f'  GRN: nodes={n_grn}, edges={g_grn.ecount()}')

        in_deg_raw = g_grn.degree(mode='in')
        in_deg_norm = [d / (n_grn - 1) for d in in_deg_raw]
        _save('grn_in_degree-raw.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'in_degree': in_deg_raw}))
        _save('grn_in_degree-norm.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'in_degree': in_deg_norm}))

        out_deg_raw = g_grn.degree(mode='out')
        out_deg_norm = [d / (n_grn - 1) for d in out_deg_raw]
        _save('grn_out_degree-raw.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'out_degree': out_deg_raw}))
        _save('grn_out_degree-norm.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'out_degree': out_deg_norm}))

        bet_r = g_grn.betweenness(directed=True, normalized=False)
        max_bet_grn = (n_grn - 1) * (n_grn - 2)
        bet_r_norm = [b / max_bet_grn for b in bet_r]
        _save('grn_betweenness-raw.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'betweenness': bet_r}))
        _save('grn_betweenness-norm.csv', pd.DataFrame({'gene': g_grn.vs['name'], 'betweenness': bet_r_norm}))

        print(f'\nDone — 8 metric files in: {save_dir}')


    def load_network_metrics(output_dir, norm=True):
        """
        Load metrics from <output_dir>/network_metrics/.
        Args:
            norm: if True load *-norm.csv (default); if False load *-raw.csv
        Returns a dict with keys:
            ppi_degree, ppi_between,
            grn_in_degree, grn_out_degree, grn_between,
            universe_ppi, universe_grn (list of gene names)
        """
        tag = 'norm' if norm else 'raw'
        save_dir = os.path.join(output_dir, 'network_metrics')

        ppi_deg_df = pd.read_csv(os.path.join(save_dir, f'ppi_degree-{tag}.csv'))
        ppi_bet_df = pd.read_csv(os.path.join(save_dir, f'ppi_betweenness-{tag}.csv'))
        grn_in_deg_df = pd.read_csv(os.path.join(save_dir, f'grn_in_degree-{tag}.csv'))
        grn_out_deg_df = pd.read_csv(os.path.join(save_dir, f'grn_out_degree-{tag}.csv'))
        grn_bet_df = pd.read_csv(os.path.join(save_dir, f'grn_betweenness-{tag}.csv'))

        ppi_degree = dict(zip(ppi_deg_df['protein'], ppi_deg_df['degree']))
        ppi_between = dict(zip(ppi_bet_df['protein'], ppi_bet_df['betweenness']))
        grn_in_degree = dict(zip(grn_in_deg_df['gene'], grn_in_deg_df['in_degree']))
        grn_out_degree = dict(zip(grn_out_deg_df['gene'], grn_out_deg_df['out_degree']))
        grn_between = dict(zip(grn_bet_df['gene'], grn_bet_df['betweenness']))

        return {
            'ppi_degree': ppi_degree,
            'ppi_between': ppi_between,
            'grn_in_degree': grn_in_degree,
            'grn_out_degree': grn_out_degree,
            'grn_between': grn_between,
            'universe_ppi': list(ppi_degree.keys()),
            'universe_grn': list(grn_in_degree.keys()),
        }


    def run_single_permutation_test(target_genes, universe, metric_dict,
                                    n_permutations=10000, random_state=42):
        """
        Single-arm permutation test for network metric enrichment.

        Parameters
        ----------
        target_genes : list
            List of gene names observed (e.g. top-DEG genes from WEDGE).
        universe : list
            Background gene universe to sample from.
        metric_dict : dict
            gene -> metric_value mapping.
        n_permutations : int
            Number of random samples.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        dict with keys:
            observed_mean  : mean metric over target_genes
            random_mean    : mean of null distribution
            random_std     : std of null distribution
            z_score        : (observed - random_mean) / random_std
            emp_pval       : empirical p-value (proportion of random >= observed)
            random_dist    : list of n_permutations null means
        """
        rng = np.random.RandomState(random_state)
        n_target = len(target_genes)

        # Filter universe to genes that have a metric
        univ_filtered = [g for g in universe if g in metric_dict]
        n_univ = len(univ_filtered)

        if n_univ == 0:
            return {
                'observed_mean': np.nan,
                'random_mean': np.nan,
                'random_std': np.nan,
                'z_score': np.nan,
                'emp_pval': np.nan,
                'random_dist': []
            }

        # Observed metric sum / mean for target genes
        obs_values = [metric_dict[g] for g in target_genes if g in metric_dict]
        observed_mean = np.mean(obs_values) if obs_values else np.nan

        # Null distribution
        null_means = []
        for _ in range(n_permutations):
            sampled = rng.choice(univ_filtered, size=n_target, replace=False)
            sampled_values = [metric_dict[g] for g in sampled]
            null_means.append(np.mean(sampled_values))

        random_mean = np.mean(null_means)
        random_std = np.std(null_means)
        z_score = (observed_mean - random_mean) / random_std if random_std > 0 else np.nan
        emp_pval = float(np.mean([m >= observed_mean for m in null_means]))

        return {
            'observed_mean': observed_mean,
            'random_mean': random_mean,
            'random_std': random_std,
            'z_score': z_score,
            'emp_pval': emp_pval,
            'random_dist': null_means
        }


    def _apply_academic_axis(ax):
        """学术风格坐标轴"""
        ax.tick_params(axis='both', which='major', direction='out',
                       top=False, right=False, width=1.5, length=6, labelsize=11)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


    def plot_null_distribution_two_arm_permutation(top_ppi_genes, top_grn_genes,
                                                   gene_list,
                                                   output_dir,
                                                   n_permutations=10000,
                                                   random_state=42,
                                                   norm=True):
        """
        每个指标单独出图，每个图有两条 KDE：
        1. 灰色  — 从全网络 (universe) 随机抽样
        2. 橙色  — 从 DEP 基因集随机抽样

        5 个指标：PPI (Degree/Betweenness) + GRN (In-Degree/Out-Degree/Betweenness)
        Loads pre-computed metrics from <output_dir>/network_metrics/.

        Args:
            norm: if True use *-norm.csv (default); if False use *-raw.csv
        """
        nets = load_network_metrics(output_dir, norm=norm)
        n_universe_ppi = len(nets['universe_ppi'])
        n_universe_grn = len(nets['universe_grn'])
        n_genelist = len(gene_list)
        print(f'Full universe — PPI: {n_universe_ppi} | GRN: {n_universe_grn}')
        print(f'DEP set       — size: {n_genelist} | '
              f'PPI overlap: {len(set(top_ppi_genes) & set(gene_list))} | '
              f'GRN overlap: {len(set(top_grn_genes) & set(gene_list))}')

        # 5 configs: PPI 2 + GRN 3
        configs = [
            # PPI (undirected): 2
            ('PPI Degree', top_ppi_genes, nets['universe_ppi'], nets['ppi_degree']),
            ('PPI Betweenness', top_ppi_genes, nets['universe_ppi'], nets['ppi_between']),
            # GRN (directed): 3 — In/Out Degree, Betweenness
            ('GRN In-Degree', top_grn_genes, nets['universe_grn'], nets['grn_in_degree']),
            ('GRN Out-Degree', top_grn_genes, nets['universe_grn'], nets['grn_out_degree']),
            ('GRN Betweenness', top_grn_genes, nets['universe_grn'], nets['grn_between']),
        ]

        results = {}
        for name, tgenes, univ_full, mdict in configs:
            print(f'\n--- {name} ---')
            res_full = run_single_permutation_test(
                tgenes, univ_full, mdict,
                n_permutations=n_permutations, random_state=random_state)
            univ = list(set(univ_full) & set(gene_list))
            res_arm = run_single_permutation_test(
                tgenes, univ, mdict,
                n_permutations=n_permutations, random_state=random_state)
            results[name] = {'full': res_full, 'arm': res_arm}
            print(f'  Full universe:  obs={res_full["observed_mean"]:.4f}, '
                  f'null={res_full["random_mean"]:.4f}±{res_full["random_std"]:.4f}, '
                  f'Z={res_full["z_score"]:.2f}, P={res_full["emp_pval"]:.4f}')
            print(f'  DEP set:        obs={res_arm["observed_mean"]:.4f}, '
                  f'null={res_arm["random_mean"]:.4f}±{res_arm["random_std"]:.4f}, '
                  f'Z={res_arm["z_score"]:.2f}, P={res_arm["emp_pval"]:.4f}')

        # ── Plot — one small figure per metric ───────────────────────────────────
        save_base = os.path.join(output_dir, 'fig/Generank')
        os.makedirs(save_base, exist_ok=True)
        plt.rcParams.update({'font.size': 11})

        color_full = '#555555'
        color_arm = '#E64B35'
        color_obs = '#3C5488'

        summary_rows = []

        for name in [c[0] for c in configs]:
            r = results[name]
            dist_full = r['full']['random_dist']
            dist_arm = r['arm']['random_dist']
            obs_full = r['full']['observed_mean']
            z_full = r['full']['z_score']
            p_full = r['full']['emp_pval']
            z_arm = r['arm']['z_score']
            p_arm = r['arm']['emp_pval']

            fig, ax = plt.subplots(figsize=(7.5, 3))

            if len(dist_full) > 20:
                sns.kdeplot(dist_full, ax=ax, color=color_full, lw=2,
                            label=f'Full universe (Z={z_full:.2f})',
                            fill=True, alpha=0.25)
            if len(dist_arm) > 20:
                sns.kdeplot(dist_arm, ax=ax, color=color_arm, lw=2,
                            label=f'DEP (Z={z_arm:.2f})',
                            fill=True, alpha=0.25)

            ax.axvline(obs_full, color=color_obs, linewidth=2.5, linestyle='--',
                       label=f'WEDGE ({obs_full:.5f})')

            def fmt_p(p):
                return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'

            ax.text(0.97, 0.65,
                    f'Full: {fmt_p(p_full)}\nDEP:  {fmt_p(p_arm)}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(name, fontsize=13,
                         # fontweight='bold'
                         )
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
            _apply_academic_axis(ax)

            plt.tight_layout()
            tag = 'norm' if norm else 'raw'
            out_path = os.path.join(save_base, f'permutation_{tag}_{name.replace(" ", "_")}.pdf')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f'Saved: {out_path}')
            plt.show()
            plt.close()

            # Collect for summary table
            for arm, arm_name in [('full', 'Full universe'), ('arm', 'DEP')]:
                row = results[name][arm].copy()
                del row['random_dist']
                summary_rows.append({
                    'Metric': name,
                    'Arm': arm_name,
                    'Observed_mean': f"{row['observed_mean']:.6f}",
                    'Null_mean': f"{row['random_mean']:.6f}",
                    'Null_std': f"{row['random_std']:.6f}",
                    'Z_score': f"{row['z_score']:.4f}",
                    'Emp_P_value': f"{row['emp_pval']:.6f}",
                })

        # Save summary table
        tag = 'norm' if norm else 'raw'
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(save_base, f'permutation_summary_{tag}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSaved summary table: {summary_path}')
        return results

# ==============================================================================
# STEP 2 — Run (after metrics are saved)
# ==============================================================================

# Step 2a: compute and save metrics (RUN ONCE)
ppi_path = os.path.join(PROJECT_ROOT, 'PPI_GRN_database', 'String_database',
                        'human_PPI_score_Stringdatabase(700up).csv')
grn_path = os.path.join(PROJECT_ROOT, 'PPI_GRN_database', 'Trrust_database',
                        'TF_filtered_human.csv')

compute_and_save_network_metrics(
    ppi_path=ppi_path,
    grn_path=grn_path,
    output_dir=OUTPUT_DIR,
    ppi_score_thresh=700
)
# Step 2b: run permutation test (uses pre-saved metrics)
TOP_PPI_GENES = [
    'PGC', 'SBSN', 'CTSE', 'POTEKP', 'FN1', 'WFDC2', 'LYZ', 'LRRC17',
    'F3', 'FAP', 'MIA', 'SIX4', 'CD248', 'RSPH3', 'TNS1'
]
TOP_GRN_GENES = [
    'ACTA2', 'FOXA1', 'BRCA1', 'DNMT1', 'ELF3', 'LUM', 'TSC22D3',
    'RB1CC1', 'DNMT3A', 'PARP1', 'ING4', 'TFAP4', 'KLF4', 'MYB', 'CDX2'
]


gene_list = DEgene_selected(matrix, path=PROJECT_ROOT).columns.tolist()


plot_null_distribution_two_arm_permutation(
    gene_list=gene_list,
    top_ppi_genes=TOP_PPI_GENES,
    top_grn_genes=TOP_GRN_GENES,
    output_dir=OUTPUT_DIR,
    n_permutations=10000,
    random_state=0,
    norm = True
)

plot_null_distribution_two_arm_permutation(
    gene_list=gene_list,
    top_ppi_genes=TOP_PPI_GENES,
    top_grn_genes=TOP_GRN_GENES,
    output_dir=OUTPUT_DIR,
    n_permutations=10000,
    random_state=0,
    norm = False
)


# def merge_network_metrics(output_dir):
#     """
#     合并 network_metrics 目录下的所有指标文件为两个文件：
#     - PPI-merge.csv: PPI degree + betweenness
#     - GRN-merge.csv: GRN in_degree + out_degree + betweenness
#     """
#     import pandas as pd
#     import os
#
#     metrics_dir = os.path.join(output_dir, 'network_metrics')
#     out_path = os.path.join(metrics_dir, 'PPI-merge.csv')
#     out_path_grn = os.path.join(metrics_dir, 'GRN-merge.csv')
#
#     # ── PPI merge ──────────────────────────────────────────────────────────────
#     ppi_degree_norm = pd.read_csv(os.path.join(metrics_dir, 'ppi_degree-norm.csv'))
#     ppi_degree_raw = pd.read_csv(os.path.join(metrics_dir, 'ppi_degree-raw.csv'))
#     ppi_between_norm = pd.read_csv(os.path.join(metrics_dir, 'ppi_betweenness-norm.csv'))
#     ppi_between_raw = pd.read_csv(os.path.join(metrics_dir, 'ppi_betweenness-raw.csv'))
#
#     ppi_merged = ppi_degree_norm[['protein']].copy()
#     ppi_merged = ppi_merged.merge(
#         ppi_degree_norm.rename(columns={'degree': 'degree_norm'}), on='protein', how='left')
#     ppi_merged = ppi_merged.merge(
#         ppi_degree_raw.rename(columns={'degree': 'degree_raw'}), on='protein', how='left')
#     ppi_merged = ppi_merged.merge(
#         ppi_between_norm.rename(columns={'betweenness': 'betweenness_norm'}), on='protein', how='left')
#     ppi_merged = ppi_merged.merge(
#         ppi_between_raw.rename(columns={'betweenness': 'betweenness_raw'}), on='protein', how='left')
#     ppi_merged = ppi_merged[['protein', 'degree_norm', 'degree_raw', 'betweenness_norm', 'betweenness_raw']]
#     ppi_merged.to_csv(out_path, index=False)
#     print(f'PPI-merge saved: {out_path} ({len(ppi_merged)} rows)')
#
#     # ── GRN merge ───────────────────────────────────────────────────────────────
#     grn_in_norm = pd.read_csv(os.path.join(metrics_dir, 'grn_in_degree-norm.csv'))
#     grn_in_raw = pd.read_csv(os.path.join(metrics_dir, 'grn_in_degree-raw.csv'))
#     grn_out_norm = pd.read_csv(os.path.join(metrics_dir, 'grn_out_degree-norm.csv'))
#     grn_out_raw = pd.read_csv(os.path.join(metrics_dir, 'grn_out_degree-raw.csv'))
#     grn_between_norm = pd.read_csv(os.path.join(metrics_dir, 'grn_betweenness-norm.csv'))
#     grn_between_raw = pd.read_csv(os.path.join(metrics_dir, 'grn_betweenness-raw.csv'))
#
#     grn_merged = grn_in_norm[['gene']].copy()
#     grn_merged = grn_merged.merge(
#         grn_in_norm.rename(columns={'in_degree': 'in_degree_norm'}), on='gene', how='left')
#     grn_merged = grn_merged.merge(
#         grn_in_raw.rename(columns={'in_degree': 'in_degree_raw'}), on='gene', how='left')
#     grn_merged = grn_merged.merge(
#         grn_out_norm.rename(columns={'out_degree': 'out_degree_norm'}), on='gene', how='left')
#     grn_merged = grn_merged.merge(
#         grn_out_raw.rename(columns={'out_degree': 'out_degree_raw'}), on='gene', how='left')
#     grn_merged = grn_merged.merge(
#         grn_between_norm.rename(columns={'betweenness': 'betweenness_norm'}), on='gene', how='left')
#     grn_merged = grn_merged.merge(
#         grn_between_raw.rename(columns={'betweenness': 'betweenness_raw'}), on='gene', how='left')
#     grn_merged = grn_merged[['gene', 'in_degree_norm', 'in_degree_raw',
#                              'out_degree_norm', 'out_degree_raw',
#                              'betweenness_norm', 'betweenness_raw']]
#     grn_merged.to_csv(out_path_grn, index=False)
#     print(f'GRN-merge saved: {out_path_grn} ({len(grn_merged)} rows)')
#
#
# merge_network_metrics(OUTPUT_DIR)

