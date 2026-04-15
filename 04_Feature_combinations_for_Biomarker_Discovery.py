# tensorboard --logdir H:/Proteomic/lighting_loggers/GraphLevelProteinGCN/
# tensorboard --logdir H:/Proteomic/lighting_loggers/GraphLevelProteinGCN2/
# http://localhost:6006/
import math

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import os
import warnings
from tqdm import tqdm
import random
import pandas as pd
import itertools
from collections import Counter
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import shap
import seaborn as sns

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# Data paths - using relative paths from project root
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Demo_data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

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
    HPV_EA_DEG = pd.read_csv(os.path.join(PPI_DIR, 'DEG_genes.csv'), index_col=0) if os.path.exists(os.path.join(PPI_DIR, 'DEG_genes.csv')) else pd.DataFrame()
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

if True:
    gene_combinations = generate_random_gene_combinations(HPV_EA_DEG, n_combinations=5000, genes_per_combo=3)
    with open(os.path.join(RESULTS_DIR, 'gene_combinations_3_unscaled.json'), 'w') as f:
        json.dump(gene_combinations, f)
    with open(os.path.join(RESULTS_DIR, 'gene_combinations_3_unscaled.json'), 'r') as f:
        loaded_gene_combinations_3genes = json.load(f)

if True:
    results_df = evaluate_gene_combinations(
        matrix=X_train,
        gene_combinations=loaded_gene_combinations_3genes,
        label=y_train,
        matrix_test=X_test,
        label_test=y_test,
        n_splits=5,
        n_repeats=3
    )
    results_df['combined_score'] = results_df.mean_test_aucs * 0.2 + results_df.mean_val_aucs * 0.8
    dump(results_df, os.path.join(RESULTS_DIR, 'results_df_unscaled.joblib'))
    loaded_results_df = load(os.path.join(RESULTS_DIR, 'results_df_unscaled.joblib'))
    results_df.to_csv(os.path.join(RESULTS_DIR, '5000_combos_3genes_unscaled.csv'))

top20_index = results_df['combined_score'].sort_values(key=lambda x: [item[0] for item in x], ascending=False).index[
              :20]
top20_combos = results_df.iloc[top20_index].genes
sort_index = results_df.combined_score.sort_values(key=lambda x: [item[0] for item in x], ascending=False).index
results_df = results_df.iloc[sort_index]

import os
os.makedirs(RESULTS_DIR, exist_ok=True)
import json
import pandas as pd
from joblib import dump, load
import os

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
n_genes = 3
combinations_file = os.path.join(RESULTS_DIR, f'gene_combinations_{n_genes}genes.json')

# 定义要测试的基因数量

gene_counts = [3, 4, 5, 7, 10]
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
                n_combinations=5000,
                genes_per_combo=n_genes,
                if_all=True
            )
        else:
            gene_combinations = generate_random_gene_combinations(
                HPV_EA_DEG,
                n_combinations=5000,
                genes_per_combo=n_genes
            )

        with open(combinations_file, 'w') as f:
            json.dump(gene_combinations, f)
        print(f"Saved gene combinations to: {combinations_file}")

    with open(combinations_file, 'r') as f:
        loaded_gene_combinations = json.load(f)
    print(f"Loaded {len(loaded_gene_combinations)} gene combinations")

    joblib_file = os.path.join(RESULTS_DIR, f'results_df_{n_genes}genes.joblib')
    csv_file = os.path.join(RESULTS_DIR, f'5000_combos_{n_genes}genes.csv')

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


gene_counts = [3, 4, 5, 7, 10]
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
    csv_file = os.path.join(RESULTS_DIR, f'5000_combos_{n_genes}genes.csv')

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
    gene_counts_list = [3, 4, 5, 7, 10]

    # Ensure the output directory exists
    save_base_path = os.path.join(OUTPUT_DIR, 'fig/fig3/Generank')
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
        top50_index = results_df[sort_col].sort_values(ascending=False).index[:40]
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

        print(f"Saved plot to: {full_save_path}")