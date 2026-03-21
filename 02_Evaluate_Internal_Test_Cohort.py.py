#%% Block 1: Imports and Setup
import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.metrics import (accuracy_score, roc_curve, auc, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
# 添加自定义库路径
sys.path.append('lib/')
try:
    from utilsdata import *
    print("Successfully imported utilsdata.")
except ImportError:
    print("Warning: 'utilsdata' module not found. Make sure path is correct.")

# 配置参数
SAVE_PATH = "H:/Proteomic/ppt/final/metric/fig/fig3/compare"
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

# 绘图设置
plt.rcParams.update({'font.size': 12})

# %% Block 2: Feature Definitions and Helper Functions
feature_genes = {
    'WEDGE': ["PGC", "DNMT1", "GATA6", "FN1", "TNS1", "CTSE", "FAP", "MFGE8", "HSPG2", "VIM", "TIMP3",
              "GAS6", "LGALS1", "NID2", "CPB2", "SERPINF1", "MMP7", "MTUS2", "CDX2",
              "FHL2", "NFIX", "TSC22D3", "KLF4", "TFAP4", "ING4", "ACTA2",
              "MYB", "ACAT2", "PARP1", "EGR2"],
    'BINN': ['ILK', 'ACTB', 'OR4D6', 'CDK4', 'POLD2', 'LIG1', 'EXOSC5', 'LDHD',
             'FXYD1', 'PARVA', 'BHLHE40', 'GFAP', 'CHUK', 'KDM3A', 'H2BC26',
             'LIMS1', 'CDK2', 'RBBP7', 'CYGB', 'TPM2', 'COL8A2', 'CA2', 'VIM',
             'COL8A1', 'TYROBP', 'CALD1', 'GPC1', 'CDK1', 'BCL2L11', 'LIMS2'],
    'POC19': ['SH3BGRL', 'FMO1', 'SMPDL3B', 'ANG', 'ABI3BP', 'TTC4', 'COL1A1',
              'H1-5', 'CTSH', 'PLA2G2A', 'F3', 'SPINK1', 'SERPINF1', 'CD6',
              'DES', 'NOLC1', 'SSX2IP', 'SAP30', 'ASRGL1', 'GNLY', 'CTR9',
              'NSL1', 'CSRP2', 'PHACTR4', 'LIG1', 'TACC3', 'TAX1BP3', 'MIS12',
              'GINS1', 'PHGDH'],
    'RF': ['MSH2', 'EXOSC5', 'TYMS', 'CDKN2A', 'MCM3', 'RBBP7', 'MCM6', 'RFC3',
           'STMN1', 'GINS2', 'CDK1', 'CDK4', 'RFC2', 'MCMBP', 'PCNA', 'RFC4',
           'MCM7', 'MCM4', 'UHRF1', 'MCM5', 'MCM2', 'DNAJC9', 'DUT', 'HAT1',
           'CDK2', 'GINS4', 'GINS3', 'NASP', 'DSN1', 'GINS1'],
    'DIABLO': ['CHTF18', 'MSH6', 'IPO9', 'FEN1', 'CDKN2A', 'PRIM2', 'WDR76',
               'RRM1', 'MCMBP', 'RBBP7', 'RFC5', 'UHRF1', 'GINS4', 'LIG1',
               'RFC3', 'GINS3', 'PCNA', 'NASP', 'RFC2', 'DNAJC9', 'RFC4',
               'TYMS', 'HAT1', 'MCM5', 'GINS1', 'MCM2', 'MCM7', 'MCM6',
               'MCM4', 'MCM3']
}
def rank_genes_by_single_accuracy(genes, X_train, y_train, ascending=True):
    scores = []
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train)
    col_map = {name: i for i, name in enumerate(X_train.columns)}

    for gene in genes:
        if gene not in col_map: continue
        idx = col_map[gene]
        X_tr_gene = X_train_full[:, idx].reshape(-1, 1)

        clf = LogisticRegressionCV(
            Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
            n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
            max_iter=1000, l1_ratios=[0.5], random_state=42  # 固定随机种子以减少波动
        )
        clf.fit(X_tr_gene, y_train)
        pred = clf.predict(X_tr_gene)
        acc = accuracy_score(y_train, pred)
        scores.append((gene, acc))

    scores.sort(key=lambda x: x[1], reverse=not ascending)
    sorted_genes = [x[0] for x in scores]
    return sorted_genes
def perform_repeated_cv(X, y, n_repeats=5, n_splits=5):
    """
    核心评估函数：
    在 Train Set 上进行 Repeated 5-Fold CV。
    返回 Acc, F1, Recall 的 Mean 和 Std。
    """
    acc_scores = []
    f1_scores = []
    rec_scores = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(n_repeats):
        # 不同的随机种子确保 Shuffle 结果不同
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # 定义模型 (参数与原代码一致)
            clf = LogisticRegressionCV(
                Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                max_iter=1000, l1_ratios=[0.5], random_state=42  # 固定随机种子以减少波动
            )

            clf.fit(X_fold_train, y_fold_train)
            y_pred = clf.predict(X_fold_val)

            # 计算指标
            acc_scores.append(accuracy_score(y_fold_val, y_pred))
            f1_scores.append(f1_score(y_fold_val, y_pred, average='weighted'))  # 或 'macro'/'binary'
            rec_scores.append(recall_score(y_fold_val, y_pred, average='weighted'))

    return {
        'acc_mean': np.mean(acc_scores), 'acc_std': np.std(acc_scores),
        'f1_mean': np.mean(f1_scores), 'f1_std': np.std(f1_scores),
        'rec_mean': np.mean(rec_scores), 'rec_std': np.std(rec_scores)
    }

# %% Block 3: Data Loading
print("Loading data...")
try:
    path = "H:/Proteomic"
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T

    # 尝试调用 DEgene_selected
    if 'DEgene_selected' in globals():
        matrix = DEgene_selected(matrix)

    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]

    # 数据切分
    if 'SplitData' in globals():
        matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    else:
        print("SplitData function not found!")
        # 这是一个 fallback，防止没有 SplitData 函数
        from sklearn.model_selection import train_test_split

        matrix_train, matrix_test, meta_train, meta_test = train_test_split(
            matrix_sub, meta_sub, test_size=0.3, stratify=meta_sub['CancerType'], random_state=42
        )

    encoder = LabelEncoder()
    label_train = encoder.fit_transform(meta_train.CancerType.values)
    label_test = encoder.transform(meta_test.CancerType.values)

    print(f"Data Loaded. Train shape: {matrix_train.shape}, Test shape: {matrix_test.shape}")
    print(f"Labels: {encoder.classes_}")

except Exception as e:
    print(f"Error loading data: {e}")




# %% Block 4: Evaluate Gene Numbers (Toggle between CV or Train/Test)
print("Evaluate Gene Numbers...")
try :
    USE_CROSS_VALIDATION = False  # False: Train训练 -> Test测试
    # ===========================================

    gene_numbers = [1, 2, 3, 5, 7, 9, 10, 12, 15, 20]
    results_list = []
    captured_feat3 = {}
    captured_feat20 = {}

    all_methods = list(feature_genes.keys())

    print(
        f"Starting Evaluation. Mode: {'Cross-Validation on Train' if USE_CROSS_VALIDATION else 'Train on Train -> Predict Test (Single Run)'}")

    for method_name in tqdm(all_methods, desc="Methods"):
        genes = feature_genes[method_name]
        available_genes = [g for g in genes if g in matrix_train.columns]

        if len(available_genes) == 0: continue

        current_genes_pool = available_genes.copy()

        if method_name in ['RF', 'DIABLO']:
            sorted_genes = rank_genes_by_single_accuracy(
                available_genes, matrix_train, label_train, ascending=True
            )
            current_genes_pool = sorted_genes

        elif method_name == 'WEDGE':
            try:
                forced_top = ["PGC", "DNMT1", "GATA6"]
                valid_forced = [g for g in forced_top if g in available_genes]
                remain = [g for g in available_genes if g not in valid_forced]

                if len(remain) > 0:
                    X_rem = matrix_train.loc[:, remain]
                    scaler_tmp = StandardScaler()
                    X_rem_s = scaler_tmp.fit_transform(X_rem)
                    lr = LogisticRegressionCV(
                        Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                        n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                        max_iter=1000, l1_ratios=[0.5], random_state=42  # 固定随机种子以减少波动
                    )
                    lr.fit(X_rem_s, label_train)
                    imp = np.abs(lr.coef_[0])
                    sorted_rem = [g for g, _ in sorted(zip(remain, imp), key=lambda x: x[1], reverse=True)]
                    current_genes_pool = valid_forced + sorted_rem
                else:
                    current_genes_pool = valid_forced
            except Exception as e:
                print(f"WEDGE optimization failed: {e}")

        method_data = {'method': method_name}

        # --- 2. 循环评估不同数量特征 ---
        for n in gene_numbers:
            # 如果基因不够，填充空值
            if n > len(current_genes_pool):
                method_data[f'acc_{n}'] = np.nan
                method_data[f'acc_{n}_std'] = np.nan
                method_data[f'genes_{n}'] = ""
                continue

            selected_genes = current_genes_pool[:n]

            # 记录当前组合的基因列表
            method_data[f'genes_{n}'] = ",".join(selected_genes)

            if n == 3: captured_feat3[method_name] = selected_genes
            if n == 20: captured_feat20[method_name] = selected_genes

            # 准备数据
            X_train_sub = matrix_train.loc[:, selected_genes]

            if USE_CROSS_VALIDATION:
                # === 模式 A: Train 集内部 CV (n_repeats=1) ===
                metrics = perform_repeated_cv(X_train_sub, label_train, n_repeats=1, n_splits=5)
                acc = metrics['acc_mean']
                acc_std = metrics['acc_std']

            else:
                # === 模式 B: Train 训练 -> Test 测试 (单次运行) ===
                X_test_sub = matrix_test.loc[:, selected_genes]

                # 标准化 (Train fit, Test transform)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_sub)
                X_te_s = scaler.transform(X_test_sub)

                # 单次训练
                clf = LogisticRegressionCV(
                    Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                    n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                    max_iter=1000, l1_ratios=[0.5], random_state=42  # 固定随机种子以减少波动
                )
                clf.fit(X_tr_s, label_train)
                y_pred = clf.predict(X_te_s)

                acc = accuracy_score(label_test, y_pred)

            # 记录结果
            method_data[f'acc_{n}'] = acc

        results_list.append(method_data)

    # --- 生成 DataFrame 并保存 CSV ---
    accuracy_df = pd.DataFrame(results_list)

    # 构建保存路径
    csv_filename = f"{SAVE_PATH}/gene_number_accuracy.csv"
    accuracy_df.to_csv(csv_filename, index=False)

    print("Evaluation complete.")
    print(f"Results (including gene lists) saved to: {csv_filename}")
    print(accuracy_df.head())
except:
    print("block 4 failed")


# %% Block 5: plot_gene_number_accuracy_custom (Final Academic Style)
def plot_gene_number_accuracy_custom(accuracy_df, gene_numbers, save_path, is_cv_mode=True):
    plt.close('all')
    # 设置画布大小
    plt.figure(figsize=(8, 4))

    # 1. 方法顺序与颜色
    target_order = ['WEDGE', 'BINN', 'POC19', 'RF', 'DIABLO']
    existing_methods = accuracy_df['method'].unique().tolist()
    methods_to_plot = [m for m in target_order if m in existing_methods]
    for m in existing_methods:
        if m not in methods_to_plot: methods_to_plot.append(m)

    # 2. 循环绘图
    for i, method in enumerate(methods_to_plot):
        row = accuracy_df[accuracy_df['method'] == method].iloc[0]
        color = colors[i % len(colors)]
        means = [row.get(f'acc_{n}', np.nan) for n in gene_numbers]

        valid_points = []
        for j, val in enumerate(means):
            if not np.isnan(val):
                valid_points.append((gene_numbers[j], val))

        if not valid_points: continue
        xs, ys = zip(*valid_points)
        xs, ys = np.array(xs), np.array(ys)

        is_wedge = (method == 'WEDGE')
        lw = 3.5
        zorder = 10 if is_wedge else 5

        # 绘制折线
        plt.plot(xs, ys, marker='o', linewidth=lw, markersize=8,
                 label=method, color=color, alpha=1.0, zorder=zorder)

        # WEDGE 显示数值
        if is_wedge:
            for x, y in zip(xs, ys):
                # 数值位置微调
                plt.text(x, y + 0.012, f'{y:.2f}', ha='center', va='bottom',
                         fontsize=10, color=color,  zorder=zorder + 1)

    # 3. 标题与标签 (学术风格)
    academic_title = "Diagnostic Performance with Varying Protein Signature Sizes"
    plt.title(academic_title, fontsize=15, pad=15)

    plt.xlabel('Number of Proteins', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # 4. 刻度线设置 (关键修改)
    # direction='out': 刻度线朝外
    # top=False, right=False: 仅显示左边和下边的刻度标记
    plt.tick_params(axis='both', which='major', direction='out',
                    top=False, right=False,
                    width=1.5, length=6, labelsize=12)

    plt.xticks(gene_numbers)
    plt.grid(False)  # 无网格

    # 5. 坐标轴边框 (Spines)
    # 保持四个边框可见，形成一个封闭的 Box，但只有左/下有刻度
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 6. Y轴范围
    plt.ylim(0.65, 1.005)

    # 7. 图例设置 (恢复边框)
    # frameon=True: 显示边框
    # edgecolor='black': 黑色边框
    # framealpha=1: 不透明背景，防止遮挡线条时看不清
    plt.legend(loc='lower right', fontsize=11,
               frameon=True, edgecolor='black', framealpha=0.9, fancybox=False)

    plt.tight_layout()

    if save_path:
        filename = "fig3_diagnostic_performance.pdf"
        full_path = f"{save_path}/{filename}"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {full_path}")

    plt.show()


# 运行
plot_gene_number_accuracy_custom(accuracy_df, gene_numbers, SAVE_PATH, is_cv_mode=USE_CROSS_VALIDATION)
# %% Block 6: Final Evaluation on Fixed 2-Gene Combinations for Different Methods (Final Corrected Style)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, roc_auc_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import torch


    # --- 1. 辅助绘图样式函数 (Block 5 风格) ---
    def apply_academic_axis(ax):
        ax.tick_params(axis='both', which='major', direction='out',
                       top=False, right=False, width=1.5, length=6, labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


    # --- 2. 核心逻辑函数 (保持不变) ---
    def train_fold(X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = LogisticRegressionCV(
            Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
            n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
            max_iter=1000, l1_ratios=[0.5], random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        result = [np.concatenate([clf.intercept_, clf.coef_.ravel()])]
        y_test_scores = clf.predict_proba(X_test_scaled)

        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_test_onehot = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
        test_aucs = []
        for i in range(y_test_onehot.shape[1]):
            try:
                auc_test = roc_auc_score(y_test_onehot[:, i], y_test_scores[:, i])
                test_aucs.append(auc_test)
            except:
                test_aucs.append(np.nan)
        return y_test, y_test_scores, test_aucs, result


    def enhanced_evaluate_test_gene_combinations(matrix, label_train, matrix_test, label_test,
                                                 gene_combinations, gene_names, n_repeats=5):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()

        results = []
        predictions_dict = {name: {'test_actual': [], 'test_scores': []} for name in gene_names}

        for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations),
                                     desc="Gene combinations"):
            if combo_idx >= len(gene_names): continue
            gene_name = gene_names[combo_idx]
            X = matrix.loc[:, genes]
            X_test_sub = matrix_test.loc[:, genes].values

            fold_metrics = {'aucs': [], 'precision_0': [], 'precision_1': [], 'recall_0': [], 'recall_1': [],
                            'f1_0': [], 'f1_1': [], 'accuracy': []}

            for _ in range(n_repeats):
                y_act, y_scr, t_aucs, _ = train_fold(X, label_train, X_test_sub, label_test)
                if len(y_scr.shape) == 2 and y_scr.shape[1] == 2: y_scr = y_scr[:, 1]
                y_act, y_scr = y_act.ravel(), y_scr.ravel()

                predictions_dict[gene_name]['test_actual'].extend(y_act)
                predictions_dict[gene_name]['test_scores'].extend(y_scr)

                y_pred = (y_scr > 0.5).astype(int)
                try:
                    p = precision_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    r = recall_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    f = f1_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    acc = accuracy_score(y_act, y_pred)
                except:
                    p, r, f = [0, 0], [0, 0], [0, 0]
                    acc = 0.0

                fold_metrics['aucs'].append(t_aucs)
                fold_metrics['accuracy'].append(acc)
                for i, m_list in enumerate([fold_metrics['precision_0'], fold_metrics['precision_1']]): m_list.append(
                    p[i])
                for i, m_list in enumerate([fold_metrics['recall_0'], fold_metrics['recall_1']]): m_list.append(r[i])
                for i, m_list in enumerate([fold_metrics['f1_0'], fold_metrics['f1_1']]): m_list.append(f[i])

            res = {'genes': genes}
            for k, v in fold_metrics.items():
                res[f'test_{k}_mean'] = np.mean(v)
                res[f'test_{k}_std'] = np.std(v)
            results.append(res)
        results_fm = pd.DataFrame(results.copy())
        results_fm.index = gene_names
        return pd.DataFrame(results_fm), pd.DataFrame(results), predictions_dict
    # --- 3. 绘图函数 ---

    def plot_roc_curves(predictions_dict, figsize=(5, 4), save_path=None, filename=None):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 14})
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)

        test_aucs = {}
        for i, (name, data) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(data['test_actual'], data['test_scores'])
            auc_val = auc(fpr, tpr)
            test_aucs[name] = auc_val
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5, alpha=0.9,
                    label=f'{name} (AUC = {auc_val:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Test Cohort ROC Curves', fontsize=18)

        apply_academic_axis(ax)
        ax.legend(loc="lower right", fontsize=11, frameon=True,
                  edgecolor='black', framealpha=0.9, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_roc_curves.pdf" if filename else "roc_curves.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()
        return test_aucs


    def plot_prf_metrics(detailed_results, gene_names, class_label, dataset_type='test', save_path=None, filename=None,
                         figsize=(7, 5)):
        plt.close('all')
        metrics = ['precision', 'recall', 'f1']
        n_metrics = len(metrics)
        n_models = len(gene_names)
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.8 / n_models
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        for i, gene_name in enumerate(gene_names):
            if i >= len(detailed_results): continue
            values = []
            for metric in metrics:
                prefix = f"{dataset_type}_{metric}_{class_label}"
                values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))

            pos = np.arange(n_metrics) + i * bar_width
            bars = ax.bar(pos, values, bar_width, alpha=0.8, color=colors[i % len(colors)], label=gene_name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8)

        ax.set_xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylim(0, 1.15)

        if class_label == 0:
            title_text = "Test Cohort PRF Metrics (Class 0, HPV-CA)"
        elif class_label == 1:
            title_text = "Test Cohort PRF Metrics (Class 1, GAS)"
        else:
            title_text = f"Test Cohort PRF Metrics (Class {class_label})"
        ax.set_title(title_text, fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models, fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_prf_class_{class_label}.pdf" if filename else f"prf_class_{class_label}.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_accuracy(detailed_results, gene_names, dataset_type='test', save_path=None, filename=None, figsize=(7, 5)):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        n_models = len(gene_names)
        bar_width = 0.6
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        valid_idxs = [i for i in range(len(gene_names)) if i < len(detailed_results)]
        valid_names = [gene_names[i] for i in valid_idxs]
        values = [detailed_results.iloc[i].get(f"{dataset_type}_accuracy_mean", 0) for i in valid_idxs]

        for i, (name, val) in enumerate(zip(valid_names, values)):
            ax.bar(i, val, bar_width, alpha=0.8, color=colors[i % len(colors)], label=name)
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(np.arange(len(valid_names)))
        ax.set_xticklabels(valid_names, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
        ax.set_ylim(0, 1.15)
        ax.set_title('Test Cohort Performance', fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(valid_names), fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_accuracy.pdf" if filename else "accuracy.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    # ==========================================
    # 4. 修改后的 Confusion Matrix 绘图函数
    #    (完全匹配你要求的样式)
    # ==========================================
    def plot_confusion_matrix_batch(y_true, y_pred, display_labels, title, save_path=None, filename=None):
        # 1. 强制使用 target_indices=[0, 1] 确保顺序
        target_indices = [0, 1]

        cm = confusion_matrix(y_true, y_pred, labels=target_indices)

        # 2. 归一化
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # 3. 绘图设置
        plt.close('all')
        plt.figure(figsize=(4.5, 3.5))
        plt.rcParams.update({'font.size': 14})

        # 绘制热力图
        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=display_labels,
                         yticklabels=display_labels,
                         annot_kws={"size": 16})  # 批量生成时通常隐藏colorbar更简洁，如果需要可改为True

        # 4. 关键样式修改：旋转 Y 轴标签
        ax.set_xticklabels(display_labels, rotation=0)  # 水平
        ax.set_yticklabels(display_labels, rotation=90, va='center')  # 垂直 + 居中

        plt.title(title, fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        # 5. 保存
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 确保文件名后缀正确
            fname = filename if filename else title.replace(" ", "_")
            if not fname.endswith('.pdf'):
                fname += '.pdf'

            full_path = os.path.join(save_path, fname)
            plt.savefig(full_path, bbox_inches='tight')
            plt.close()  # 批量处理时关闭图形释放内存
        else:
            plt.show()


    def batch_evaluate_confusion_matrices(matrix_train, label_train, matrix_test, label_test,
                                          gene_combinations, gene_names, class_labels, save_path):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()
        label_train, label_test = label_train.ravel(), label_test.ravel()
        summary = []

        print(f"Batch evaluation for {len(gene_names)} combinations...")
        for idx, (genes, name) in tqdm(enumerate(zip(gene_combinations, gene_names)), total=len(gene_names)):
            X_tr = matrix_train.loc[:, genes].values
            X_te = matrix_test.loc[:, genes].values
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegressionCV(Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                                       n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                                       max_iter=1000, l1_ratios=[0.5], random_state=42)
            try:
                clf.fit(X_tr_s, label_train)
                y_tr_pred = clf.predict(X_tr_s)
                y_te_pred = clf.predict(X_te_s)

                summary.append({'Gene_Set': name, 'Train_Acc': accuracy_score(label_train, y_tr_pred),
                                'Test_Acc': accuracy_score(label_test, y_te_pred)})

                # 调用修改后的绘图函数
                plot_confusion_matrix_batch(
                    y_true=label_train,
                    y_pred=y_tr_pred,
                    display_labels=class_labels,
                    title=f"{name}",
                    save_path=save_path,
                    filename=f"{name}_train_cm"
                )

                plot_confusion_matrix_batch(
                    y_true=label_test,
                    y_pred=y_te_pred,
                    display_labels=class_labels,
                    title=f"{name} (Test Cohort)",
                    save_path=save_path,
                    filename=f"{name}_test_cm"
                )
            except Exception as e:
                print(f"Error {name}: {e}")
        return pd.DataFrame(summary)


    # --- Execution Logic ---
    RF_gene = ["RFC3", "EXOSC5"]
    POC_gene = ["SH3BGRL", "FMO1"]
    WEDGE = ["PGC", "DNMT1"]
    Diablo_gene = ["WDR76", "RRM1"]
    BINN_gene = ["ILK", "ACTB"]

    gene_names = ["WEGDE", "BINN", 'POC-19', 'RF', 'Diablo']
    gene_comb = [WEDGE, BINN_gene, POC_gene, RF_gene, Diablo_gene]

    # 1. Evaluate
    formatted_df, df_results, predictions = enhanced_evaluate_test_gene_combinations(
        matrix_train, label_train, matrix_test, label_test,
        gene_combinations=gene_comb, gene_names=gene_names
    )
    formatted_df.to_csv("H:/Proteomic/ppt/final/metric/fig/fig4/formatted_results.csv", index=True)
    save_fig_path = "H:/Proteomic/ppt/final/metric/fig/fig4/"
    os.makedirs(save_fig_path, exist_ok=True)
    # 2. Plots
    test_aucs = plot_roc_curves(predictions, save_path=save_fig_path, filename="new")

    plot_prf_metrics(df_results, gene_names, class_label=0, dataset_type='test',
                     save_path=save_fig_path, filename="new")
    plot_prf_metrics(df_results, gene_names, class_label=1, dataset_type='test',
                     save_path=save_fig_path, filename="new")

    plot_accuracy(df_results, gene_names, dataset_type='test',
                  save_path=save_fig_path, filename="new")

    # 3. Confusion Matrices
    save_dir_cm = "H:/Proteomic/ppt/final/metric/fig/fig4/cm_batch"

    # 定义显示的标签
    display_labels = ['HPV_CA', 'GAS']

    acc_summary = batch_evaluate_confusion_matrices(
        matrix_train=matrix_train, label_train=label_train,
        matrix_test=matrix_test, label_test=label_test,
        gene_combinations=gene_comb, gene_names=gene_names,
        class_labels=display_labels,
        save_path=save_dir_cm
    )

    print("Block 6 execution complete.")

except Exception as e:
    print(f"Error in Block 6: {e}")
    import traceback

    traceback.print_exc()

# %% Block 6.1: SHAP for WEDGE
# 忽略 sklearn 的 FutureWarning 以保持输出整洁
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def evaluate_shap_feature_importance(matrix_train, label_train, matrix_test,
                                     gene_list, gene_set_name,
                                     save_path=None, filename_suffix="shap_summary"):
    """
    训练模型并生成 SHAP 摘要图 (Beeswarm plot) - 修正版
    """
    print(f"Generating SHAP plot for: {gene_set_name} {gene_list}...")

    # 1. 准备数据
    X_train = matrix_train.loc[:, gene_list].values
    X_test = matrix_test.loc[:, gene_list].values

    # 确保标签是数值型
    if hasattr(label_train, 'cpu'): label_train = label_train.cpu().numpy()
    label_train = label_train.ravel()

    # 2. 标准化 (必须与之前的评估保持一致)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 训练模型 (保持参数一致)
    clf = LogisticRegressionCV(
        Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
        n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
        max_iter=1000, l1_ratios=[0.5], random_state=42
    )
    clf.fit(X_train_scaled, label_train)

    # 4. 创建解释器 (LinearExplainer 适用于 LogisticRegression)
    # FIX: 将 feature_dependence="independent" 改为 feature_perturbation="interventional"
    # 或者直接省略，因为当提供了 masker (X_train_scaled) 时，默认为 interventional
    explainer = shap.LinearExplainer(clf, X_train_scaled, feature_perturbation="interventional")

    # 计算测试集的 SHAP 值
    shap_values = explainer.shap_values(X_test_scaled)

    # 5. 处理 SHAP 值的形状
    # 对于二分类，shap_values 通常是一个列表 [class0_shap, class1_shap] 或 array
    # 我们通常关注正类 (Class 1) 的贡献
    # 注意：shap 版本不同，返回值结构可能略有不同
    if isinstance(shap_values, list):
        # 如果是列表，通常 index 1 是正类 (GAS)
        shap_values_to_plot = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # (samples, features, classes)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        # 如果是 (samples, features)，直接使用
        shap_values_to_plot = shap_values

    # 6. 绘图配置
    plt.close('all')  # 清理旧图
    plt.figure(figsize=(2, 4))
    plt.rcParams.update({'font.size': 14})

    # 绘制 Summary Plot (Beeswarm)
    # feature_names 使用原始基因名
    shap.summary_plot(shap_values_to_plot, X_test_scaled, feature_names=gene_list, show=False, plot_type="dot")

    plt.title(f"SHAP Summary: {gene_set_name}", fontsize=16)

    # 7. 保存
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        full_path = os.path.join(save_path, f"{gene_set_name}_{filename_suffix}.pdf")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"SHAP plot saved to: {full_path}")
        plt.show()  # 保存后也展示一下
    else:
        plt.show()
# 定义 WEDGE 基因
WEDGE_genes = ["PGC", "DNMT1"]
save_dir_fig4 = "H:/Proteomic/ppt/final/metric/fig/fig4/"
# 调用函数
evaluate_shap_feature_importance(
    matrix_train=matrix_train,
    label_train=label_train,
    matrix_test=matrix_test,
    gene_list=WEDGE_genes,
    gene_set_name="WEDGE",
    save_path=save_dir_fig4
)

# %% Block 7: Final Evaluation on Fixed 2-Gene or 1-Gene
try:
    ACAT2 = ["ACAT2"]
    PGC = ["PGC"]
    DNMT1 = ["DNMT1"]
    DNMT1_PGC = ["DNMT1","PGC"]
    ACAT2_PGC = ["ACAT2","PGC"]
    DNMT1_ACAT2 = ["DNMT1","ACAT2"]

    gene_names = ["ACAT2", "PGC", "DNMT1", "DNMT1 & PGC", "ACAT2 & PGC", "DNMT1 & ACAT2"]
    gene_comb = [ACAT2, PGC, DNMT1, DNMT1_PGC,ACAT2_PGC,DNMT1_ACAT2]
    formatted_df, df_results, predictions = enhanced_evaluate_test_gene_combinations(matrix_train, label_train,
                                                                                     matrix_test, label_test,
                                                                                     gene_combinations=gene_comb,
                                                                                     gene_names=gene_names)
    formatted_df
    test_aucs = plot_roc_curves(predictions, save_path="H:/Proteomic/ppt/final/metric/fig/S3fig/new/", filename="new",figsize=(5, 5))
    plot_prf_metrics(df_results, gene_names, class_label=0, dataset_type='test',
                     save_path="H:/Proteomic/ppt/final/metric/fig/S3fig/new/", filename="new",figsize=(9.5, 4))
    plot_prf_metrics(df_results, gene_names, class_label=1, dataset_type='test',
                     save_path="H:/Proteomic/ppt/final/metric/fig/S3fig/new/", filename="new",figsize=(9.5, 4))
    plot_accuracy(df_results, gene_names, dataset_type='test', save_path="H:/Proteomic/ppt/final/metric/fig/S3fig/new/",
                  filename="new",figsize=(12, 4))
    save_dir_cm = "H:/Proteomic/ppt/final/metric/fig/S3fig/new/cm_batch"

    acc_summary = batch_evaluate_confusion_matrices(
        matrix_train=matrix_train,
        label_train=label_train,  # 确保这里是数值型的 array (例如 [0, 1, 0...])
        matrix_test=matrix_test,
        label_test=label_test,  # 确保这里是数值型的 array
        gene_combinations=gene_comb,
        gene_names=gene_names,
        class_labels=['HPV_CA', 'GAS'],  # 混淆矩阵显示的标签名
        save_path=save_dir_cm
    )
except:
    print("Error in Block 7")

# %% Block 8: External Validation Evaluation (Final Style)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, roc_auc_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegressionCV
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import torch

    # ==============================================================================
    # 0. 数据准备 (Data Loading)
    # ==============================================================================
    print("Loading Data for External Validation...")
    path = "H:/Proteomic"

    # 1. 加载内部训练数据 (用于训练模型)
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    # 简单筛选 (假设已经做过DE筛选，或者直接用全量)
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]

    # 划分数据拿到 matrix_train (这里我们只需要 Train 部分)
    # 注意：这里需要保证 SplitData 逻辑一致，或者手动划分，此处假设 SplitData 已定义
    # 如果没有 SplitData 函数，这里简写逻辑：
    from sklearn.model_selection import train_test_split

    # 模拟 SplitData 的行为，或者你可以直接引用之前的 matrix_train
    # 这里为了代码独立性，我假设 matrix_train 已经存在于上下文中。
    # 如果是新运行，请取消下面几行的注释并确保逻辑正确：
    # X_train_raw, _, y_train_raw, _ = train_test_split(matrix_sub, meta_sub, test_size=0.2, stratify=meta_sub.CancerType, random_state=42)
    # matrix_train = X_train_raw
    # meta_train = y_train_raw

    # 2. 加载外部验证数据 (用于测试)
    matrix_external = pd.read_csv(f'{path}/data/hGCN/expr_external_GAS_HPVCA.csv', index_col=0).T
    meta_external = pd.read_csv(f'{path}/data/hGCN/meta_external_GAS_HPVCA.csv', index_col=0)

    # 3. 标签编码
    encoder = LabelEncoder()
    # 必须用内部数据的标签拟合，保证 0/1 对应关系一致
    label_train = encoder.fit_transform(meta_train.CancerType)
    label_external = encoder.transform(meta_external.CancerType)  # Transform 外部数据

    print(f"Train Set: {matrix_train.shape}, External Set: {matrix_external.shape}")


    # --- 1. 辅助绘图样式函数 ---
    def apply_academic_axis(ax):
        ax.tick_params(axis='both', which='major', direction='out',
                       top=False, right=False, width=1.5, length=6, labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


    # --- 2. 核心逻辑函数 ---
    def train_fold(X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # 使用训练集的参数标准化外部数据

        clf = LogisticRegressionCV(
            Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
            n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
            max_iter=2000, l1_ratios=[0.5], random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        result = [np.concatenate([clf.intercept_, clf.coef_.ravel()])]
        y_test_scores = clf.predict_proba(X_test_scaled)

        onehot_encoder = OneHotEncoder(sparse_output=False)
        # 处理可能出现的单类别情况
        if len(np.unique(y_test)) < 2:
            # 如果外部验证集很小，可能缺类，这里做个简单容错
            y_test_onehot = np.zeros((len(y_test), 2))
            y_test_onehot[np.arange(len(y_test)), y_test] = 1
        else:
            y_test_onehot = onehot_encoder.fit_transform(y_test.reshape(-1, 1))

        test_aucs = []
        # 确保有两个类别的分数
        if y_test_scores.shape[1] == 2:
            try:
                # 关注 Class 1 (GAS) 的 AUC，或者多分类平均
                # 这里通常取 macro 或者针对某一类，这里逻辑保持不变
                for i in range(y_test_onehot.shape[1]):
                    auc_test = roc_auc_score(y_test_onehot[:, i], y_test_scores[:, i])
                    test_aucs.append(auc_test)
            except:
                test_aucs.append(np.nan)
        return y_test, y_test_scores, test_aucs, result


    def enhanced_evaluate_test_gene_combinations(matrix, label_train, matrix_test, label_test,
                                                 gene_combinations, gene_names,
                                                 n_repeats=1):  # External 不需要 repeat 多次，结果是固定的，设为1
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()

        results = []
        predictions_dict = {name: {'test_actual': [], 'test_scores': []} for name in gene_names}

        for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations),
                                     desc="Gene combinations"):
            if combo_idx >= len(gene_names): continue
            gene_name = gene_names[combo_idx]

            # 提取数据
            X = matrix.loc[:, genes]
            X_test_sub = matrix_test.loc[:, genes].values

            fold_metrics = {'aucs': [], 'precision_0': [], 'precision_1': [], 'recall_0': [], 'recall_1': [],
                            'f1_0': [], 'f1_1': [], 'accuracy': []}

            for _ in range(n_repeats):
                y_act, y_scr, t_aucs, _ = train_fold(X, label_train, X_test_sub, label_test)
                if len(y_scr.shape) == 2 and y_scr.shape[1] == 2: y_scr = y_scr[:, 1]
                y_act, y_scr = y_act.ravel(), y_scr.ravel()

                predictions_dict[gene_name]['test_actual'].extend(y_act)
                predictions_dict[gene_name]['test_scores'].extend(y_scr)

                y_pred = (y_scr > 0.5).astype(int)
                try:
                    p = precision_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    r = recall_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    f = f1_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    acc = accuracy_score(y_act, y_pred)
                except:
                    p, r, f = [0, 0], [0, 0], [0, 0]
                    acc = 0.0

                fold_metrics['aucs'].append(t_aucs)
                fold_metrics['accuracy'].append(acc)
                for i, m_list in enumerate([fold_metrics['precision_0'], fold_metrics['precision_1']]): m_list.append(
                    p[i])
                for i, m_list in enumerate([fold_metrics['recall_0'], fold_metrics['recall_1']]): m_list.append(r[i])
                for i, m_list in enumerate([fold_metrics['f1_0'], fold_metrics['f1_1']]): m_list.append(f[i])

            res = {'genes': genes}
            for k, v in fold_metrics.items():
                res[f'test_{k}_mean'] = np.mean(v)
                res[f'test_{k}_std'] = np.std(v)
            results.append(res)

        return pd.DataFrame(), pd.DataFrame(results), predictions_dict


    # --- 3. 绘图函数 (保持不变) ---
    def plot_roc_curves(predictions_dict, figsize=(5, 4), save_path=None, filename=None):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 14})
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)

        test_aucs = {}
        for i, (name, data) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(data['test_actual'], data['test_scores'])
            auc_val = auc(fpr, tpr)
            test_aucs[name] = auc_val
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5, alpha=0.9,
                    label=f'{name} (AUC = {auc_val:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('External Cohort ROC Curves', fontsize=18)  # 修改标题

        apply_academic_axis(ax)
        ax.legend(loc="lower right", fontsize=11, frameon=True,
                  edgecolor='black', framealpha=0.9, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_roc_curves.pdf" if filename else "roc_curves.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()
        return test_aucs


    def plot_prf_metrics(detailed_results, gene_names, class_label, dataset_type='test', save_path=None, filename=None,
                         figsize=(7, 5)):
        plt.close('all')
        metrics = ['precision', 'recall', 'f1']
        n_metrics = len(metrics)
        n_models = len(gene_names)
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.8 / n_models
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        for i, gene_name in enumerate(gene_names):
            if i >= len(detailed_results): continue
            values = []
            for metric in metrics:
                prefix = f"{dataset_type}_{metric}_{class_label}"
                values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))

            pos = np.arange(n_metrics) + i * bar_width
            bars = ax.bar(pos, values, bar_width, alpha=0.8, color=colors[i % len(colors)], label=gene_name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8)

        ax.set_xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylim(0, 1.15)

        # 动态修改标题
        if class_label == 0:
            title_text = "External Cohort PRF Metrics (Class 0, HPV-CA)"
        elif class_label == 1:
            title_text = "External Cohort PRF Metrics (Class 1, GAS)"
        else:
            title_text = f"External Cohort PRF Metrics (Class {class_label})"
        ax.set_title(title_text, fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models, fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_prf_class_{class_label}.pdf" if filename else f"prf_class_{class_label}.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_accuracy(detailed_results, gene_names, dataset_type='test', save_path=None, filename=None, figsize=(7, 5)):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        n_models = len(gene_names)
        bar_width = 0.6
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        valid_idxs = [i for i in range(len(gene_names)) if i < len(detailed_results)]
        valid_names = [gene_names[i] for i in valid_idxs]
        values = [detailed_results.iloc[i].get(f"{dataset_type}_accuracy_mean", 0) for i in valid_idxs]

        for i, (name, val) in enumerate(zip(valid_names, values)):
            ax.bar(i, val, bar_width, alpha=0.8, color=colors[i % len(colors)], label=name)
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(np.arange(len(valid_names)))
        ax.set_xticklabels(valid_names, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
        ax.set_ylim(0, 1.15)
        ax.set_title('External Cohort Performance', fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(valid_names), fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_accuracy.pdf" if filename else "accuracy.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_confusion_matrix_batch(y_true, y_pred, display_labels, title, save_path=None, filename=None):
        target_indices = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=target_indices)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        plt.close('all')
        plt.figure(figsize=(4.5, 3.5))
        plt.rcParams.update({'font.size': 14})

        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=display_labels,
                         yticklabels=display_labels,
                         annot_kws={"size": 16})

        ax.set_xticklabels(display_labels, rotation=0)
        ax.set_yticklabels(display_labels, rotation=90, va='center')

        plt.title(title, fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        if save_path:
            if not os.path.exists(save_path): os.makedirs(save_path)
            fname = filename if filename else title.replace(" ", "_")
            if not fname.endswith('.pdf'): fname += '.pdf'
            full_path = os.path.join(save_path, fname)
            plt.savefig(full_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def batch_evaluate_confusion_matrices(matrix_train, label_train, matrix_test, label_test,
                                          gene_combinations, gene_names, class_labels, save_path):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()
        label_train, label_test = label_train.ravel(), label_test.ravel()

        print(f"Batch CM evaluation on External Cohort...")
        for idx, (genes, name) in tqdm(enumerate(zip(gene_combinations, gene_names)), total=len(gene_names)):
            # 训练数据 (Internal)
            X_tr = matrix_train.loc[:, genes].values
            # 测试数据 (External)
            X_te = matrix_test.loc[:, genes].values

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegressionCV(Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                                       n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                                       max_iter=1000, l1_ratios=[0.5], random_state=42)
            try:
                clf.fit(X_tr_s, label_train)
                # y_tr_pred = clf.predict(X_tr_s) # 外部验证通常不画训练集混淆矩阵，如果需要可保留
                y_te_pred = clf.predict(X_te_s)

                # 只画外部验证集的混淆矩阵
                plot_confusion_matrix_batch(
                    y_true=label_test,
                    y_pred=y_te_pred,
                    display_labels=class_labels,
                    title=f"{name} (External Cohort)",
                    save_path=save_path,
                    filename=f"{name}_external_cm"
                )
            except Exception as e:
                print(f"Error {name}: {e}")


    # --- Execution Logic ---

    # 定义基因集 (WEDGE)
    WEDGE = ["PGC", "DNMT1"]
    RF_gene = ["RFC3", "EXOSC5"]
    POC_gene = ["SH3BGRL", "FMO1"]
    Diablo_gene = ["WDR76", "RRM1"]
    BINN_gene = ["ILK", "ACTB"]

    gene_names = ["WEDGE", "BINN", 'POC-19', 'RF', 'Diablo']
    gene_comb = [WEDGE, BINN_gene, POC_gene, RF_gene, Diablo_gene]

    # 1. Evaluate (Test Set = Matrix External)
    # n_repeats = 1 因为外部验证集是固定的，不需要重复采样
    formatted_df, df_results, predictions = enhanced_evaluate_test_gene_combinations(
        matrix_train, label_train, matrix_external, label_external,
        gene_combinations=gene_comb, gene_names=gene_names, n_repeats=1
    )
    formatted_df.to_csv("H:/Proteomic/ppt/final/metric/fig/fig4/formatted_results_external.csv", index=True)
    save_fig_path = "H:/Proteomic/ppt/final/metric/fig/fig3/external/"
    os.makedirs(save_fig_path, exist_ok=True)

    # 2. Plots
    test_aucs = plot_roc_curves(predictions, save_path=save_fig_path, filename="External_Comparison")

    plot_prf_metrics(df_results, gene_names, class_label=0, dataset_type='test',
                     save_path=save_fig_path, filename="External_Comparison")
    plot_prf_metrics(df_results, gene_names, class_label=1, dataset_type='test',
                     save_path=save_fig_path, filename="External_Comparison")

    plot_accuracy(df_results, gene_names, dataset_type='test',
                  save_path=save_fig_path, filename="External_Comparison")

    # 3. Confusion Matrices
    save_dir_cm = os.path.join(save_fig_path, "cm_batch")
    display_labels = ['HPV_CA', 'GAS']

    batch_evaluate_confusion_matrices(
        matrix_train=matrix_train, label_train=label_train,
        matrix_test=matrix_external, label_test=label_external,
        gene_combinations=gene_comb, gene_names=gene_names,
        class_labels=display_labels,
        save_path=save_dir_cm
    )

    print("External Validation execution complete.")
except Exception as e:
    print(f"Error in External Validation: {e}")
    import traceback

    traceback.print_exc()

# for jijin
# %% Block 9.1: Final Evaluation on Fixed 1Gene Combinations for Different Methods (Final Corrected Style)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, roc_auc_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import torch


    # --- 1. 辅助绘图样式函数 (Block 5 风格) ---
    def apply_academic_axis(ax):
        ax.tick_params(axis='both', which='major', direction='out',
                       top=False, right=False, width=1.5, length=6, labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


    # --- 2. 核心逻辑函数 (保持不变) ---
    def train_fold(X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = LogisticRegressionCV(
            Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
            n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
            max_iter=1000, l1_ratios=[0.5], random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        result = [np.concatenate([clf.intercept_, clf.coef_.ravel()])]
        y_test_scores = clf.predict_proba(X_test_scaled)

        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_test_onehot = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
        test_aucs = []
        for i in range(y_test_onehot.shape[1]):
            try:
                auc_test = roc_auc_score(y_test_onehot[:, i], y_test_scores[:, i])
                test_aucs.append(auc_test)
            except:
                test_aucs.append(np.nan)
        return y_test, y_test_scores, test_aucs, result


    def enhanced_evaluate_test_gene_combinations(matrix, label_train, matrix_test, label_test,
                                                 gene_combinations, gene_names, n_repeats=5):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()

        results = []
        predictions_dict = {name: {'test_actual': [], 'test_scores': []} for name in gene_names}

        for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations),
                                     desc="Gene combinations"):
            if combo_idx >= len(gene_names): continue
            gene_name = gene_names[combo_idx]
            X = matrix.loc[:, genes]
            X_test_sub = matrix_test.loc[:, genes].values

            fold_metrics = {'aucs': [], 'precision_0': [], 'precision_1': [], 'recall_0': [], 'recall_1': [],
                            'f1_0': [], 'f1_1': [], 'accuracy': []}

            for _ in range(n_repeats):
                y_act, y_scr, t_aucs, _ = train_fold(X, label_train, X_test_sub, label_test)
                if len(y_scr.shape) == 2 and y_scr.shape[1] == 2: y_scr = y_scr[:, 1]
                y_act, y_scr = y_act.ravel(), y_scr.ravel()

                predictions_dict[gene_name]['test_actual'].extend(y_act)
                predictions_dict[gene_name]['test_scores'].extend(y_scr)

                y_pred = (y_scr > 0.5).astype(int)
                try:
                    p = precision_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    r = recall_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    f = f1_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    acc = accuracy_score(y_act, y_pred)
                except:
                    p, r, f = [0, 0], [0, 0], [0, 0]
                    acc = 0.0

                fold_metrics['aucs'].append(t_aucs)
                fold_metrics['accuracy'].append(acc)
                for i, m_list in enumerate([fold_metrics['precision_0'], fold_metrics['precision_1']]): m_list.append(
                    p[i])
                for i, m_list in enumerate([fold_metrics['recall_0'], fold_metrics['recall_1']]): m_list.append(r[i])
                for i, m_list in enumerate([fold_metrics['f1_0'], fold_metrics['f1_1']]): m_list.append(f[i])

            res = {'genes': genes}
            for k, v in fold_metrics.items():
                res[f'test_{k}_mean'] = np.mean(v)
                res[f'test_{k}_std'] = np.std(v)
            results.append(res)
        results_fm = pd.DataFrame(results.copy())
        results_fm.index = gene_names
        return pd.DataFrame(results_fm), pd.DataFrame(results), predictions_dict
    # --- 3. 绘图函数 ---

    def plot_roc_curves(predictions_dict, figsize=(5, 4), save_path=None, filename=None):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 14})
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)

        test_aucs = {}
        for i, (name, data) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(data['test_actual'], data['test_scores'])
            auc_val = auc(fpr, tpr)
            test_aucs[name] = auc_val
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5, alpha=0.9,
                    label=f'{name} (AUC = {auc_val:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Test Cohort ROC Curves', fontsize=18)

        apply_academic_axis(ax)
        ax.legend(loc="lower right", fontsize=11, frameon=True,
                  edgecolor='black', framealpha=0.9, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_roc_curves.pdf" if filename else "roc_curves.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()
        return test_aucs


    def plot_prf_metrics(detailed_results, gene_names, class_label, dataset_type='test', save_path=None, filename=None,
                         figsize=(7, 5)):
        plt.close('all')
        metrics = ['precision', 'recall', 'f1']
        n_metrics = len(metrics)
        n_models = len(gene_names)
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.8 / n_models
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        for i, gene_name in enumerate(gene_names):
            if i >= len(detailed_results): continue
            values = []
            for metric in metrics:
                prefix = f"{dataset_type}_{metric}_{class_label}"
                values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))

            pos = np.arange(n_metrics) + i * bar_width
            bars = ax.bar(pos, values, bar_width, alpha=0.8, color=colors[i % len(colors)], label=gene_name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8)

        ax.set_xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylim(0, 1.15)

        if class_label == 0:
            title_text = "Test Cohort PRF Metrics (Class 0, HPV-CA)"
        elif class_label == 1:
            title_text = "Test Cohort PRF Metrics (Class 1, GAS)"
        else:
            title_text = f"Test Cohort PRF Metrics (Class {class_label})"
        ax.set_title(title_text, fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models, fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_prf_class_{class_label}.pdf" if filename else f"prf_class_{class_label}.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_accuracy(detailed_results, gene_names, dataset_type='test', save_path=None, filename=None, figsize=(7, 5)):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        n_models = len(gene_names)
        bar_width = 0.6
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        valid_idxs = [i for i in range(len(gene_names)) if i < len(detailed_results)]
        valid_names = [gene_names[i] for i in valid_idxs]
        values = [detailed_results.iloc[i].get(f"{dataset_type}_accuracy_mean", 0) for i in valid_idxs]

        for i, (name, val) in enumerate(zip(valid_names, values)):
            ax.bar(i, val, bar_width, alpha=0.8, color=colors[i % len(colors)], label=name)
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(np.arange(len(valid_names)))
        ax.set_xticklabels(valid_names, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
        ax.set_ylim(0, 1.15)
        ax.set_title('Test Cohort Performance', fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(valid_names), fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_accuracy.pdf" if filename else "accuracy.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    # ==========================================
    # 4. 修改后的 Confusion Matrix 绘图函数
    #    (完全匹配你要求的样式)
    # ==========================================
    def plot_confusion_matrix_batch(y_true, y_pred, display_labels, title, save_path=None, filename=None):
        # 1. 强制使用 target_indices=[0, 1] 确保顺序
        target_indices = [0, 1]

        cm = confusion_matrix(y_true, y_pred, labels=target_indices)

        # 2. 归一化
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # 3. 绘图设置
        plt.close('all')
        plt.figure(figsize=(4.5, 3.5))
        plt.rcParams.update({'font.size': 14})

        # 绘制热力图
        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=display_labels,
                         yticklabels=display_labels,
                         annot_kws={"size": 16})  # 批量生成时通常隐藏colorbar更简洁，如果需要可改为True

        # 4. 关键样式修改：旋转 Y 轴标签
        ax.set_xticklabels(display_labels, rotation=0)  # 水平
        ax.set_yticklabels(display_labels, rotation=90, va='center')  # 垂直 + 居中

        plt.title(title, fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        # 5. 保存
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 确保文件名后缀正确
            fname = filename if filename else title.replace(" ", "_")
            if not fname.endswith('.pdf'):
                fname += '.pdf'

            full_path = os.path.join(save_path, fname)
            plt.savefig(full_path, bbox_inches='tight')
            plt.close()  # 批量处理时关闭图形释放内存
        else:
            plt.show()


    def batch_evaluate_confusion_matrices(matrix_train, label_train, matrix_test, label_test,
                                          gene_combinations, gene_names, class_labels, save_path):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()
        label_train, label_test = label_train.ravel(), label_test.ravel()
        summary = []

        print(f"Batch evaluation for {len(gene_names)} combinations...")
        for idx, (genes, name) in tqdm(enumerate(zip(gene_combinations, gene_names)), total=len(gene_names)):
            X_tr = matrix_train.loc[:, genes].values
            X_te = matrix_test.loc[:, genes].values
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegressionCV(Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                                       n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                                       max_iter=1000, l1_ratios=[0.5], random_state=42)
            try:
                clf.fit(X_tr_s, label_train)
                y_tr_pred = clf.predict(X_tr_s)
                y_te_pred = clf.predict(X_te_s)

                summary.append({'Gene_Set': name, 'Train_Acc': accuracy_score(label_train, y_tr_pred),
                                'Test_Acc': accuracy_score(label_test, y_te_pred)})

                # 调用修改后的绘图函数
                plot_confusion_matrix_batch(
                    y_true=label_train,
                    y_pred=y_tr_pred,
                    display_labels=class_labels,
                    title=f"{name}",
                    save_path=save_path,
                    filename=f"{name}_train_cm"
                )

                plot_confusion_matrix_batch(
                    y_true=label_test,
                    y_pred=y_te_pred,
                    display_labels=class_labels,
                    title=f"{name} (Test Cohort)",
                    save_path=save_path,
                    filename=f"{name}_test_cm"
                )
            except Exception as e:
                print(f"Error {name}: {e}")
        return pd.DataFrame(summary)


    # --- Execution Logic ---
    RF_gene = ["RFC3",]
    POC_gene = ["SH3BGRL"]
    WEDGE = ["PGC"]
    Diablo_gene = ["WDR76"]
    BINN_gene = ["ILK"]

    gene_names = ["WEGDE", "BINN", 'POC-19', 'RF', 'Diablo']
    gene_comb = [WEDGE, BINN_gene, POC_gene, RF_gene, Diablo_gene]

    # 1. Evaluate
    formatted_df, df_results, predictions = enhanced_evaluate_test_gene_combinations(
        matrix_train, label_train, matrix_test, label_test,
        gene_combinations=gene_comb, gene_names=gene_names
    )
    formatted_df.to_csv("H:/Proteomic/ppt/final/metric/fig/fig4/jijin/formatted_results.csv", index=True)
    save_fig_path = "H:/Proteomic/ppt/final/metric/fig/fig4/jijin"
    os.makedirs(save_fig_path, exist_ok=True)
    # 2. Plots
    test_aucs = plot_roc_curves(predictions, save_path=save_fig_path, filename="new")

    plot_prf_metrics(df_results, gene_names, class_label=0, dataset_type='test',
                     save_path=save_fig_path, filename="new")
    plot_prf_metrics(df_results, gene_names, class_label=1, dataset_type='test',
                     save_path=save_fig_path, filename="new")

    plot_accuracy(df_results, gene_names, dataset_type='test',
                  save_path=save_fig_path, filename="new")

    # 3. Confusion Matrices
    save_dir_cm = "H:/Proteomic/ppt/final/metric/fig/fig4/cm_batch/jijin"

    # 定义显示的标签
    display_labels = ['HPV_CA', 'GAS']

    acc_summary = batch_evaluate_confusion_matrices(
        matrix_train=matrix_train, label_train=label_train,
        matrix_test=matrix_test, label_test=label_test,
        gene_combinations=gene_comb, gene_names=gene_names,
        class_labels=display_labels,
        save_path=save_dir_cm
    )

    print("Block 6 execution complete.")

except Exception as e:
    print(f"Error in Block 6: {e}")
    import traceback

    traceback.print_exc()

# %% Block 9.2: External Validation Evaluation for 1 gene
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, roc_auc_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegressionCV
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import torch

    # ==============================================================================
    # 0. 数据准备 (Data Loading)
    # ==============================================================================
    print("Loading Data for External Validation...")
    path = "H:/Proteomic"

    # 1. 加载内部训练数据 (用于训练模型)
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    # 简单筛选 (假设已经做过DE筛选，或者直接用全量)
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]

    # 划分数据拿到 matrix_train (这里我们只需要 Train 部分)
    # 注意：这里需要保证 SplitData 逻辑一致，或者手动划分，此处假设 SplitData 已定义
    # 如果没有 SplitData 函数，这里简写逻辑：
    from sklearn.model_selection import train_test_split

    # 模拟 SplitData 的行为，或者你可以直接引用之前的 matrix_train
    # 这里为了代码独立性，我假设 matrix_train 已经存在于上下文中。
    # 如果是新运行，请取消下面几行的注释并确保逻辑正确：
    # X_train_raw, _, y_train_raw, _ = train_test_split(matrix_sub, meta_sub, test_size=0.2, stratify=meta_sub.CancerType, random_state=42)
    # matrix_train = X_train_raw
    # meta_train = y_train_raw

    # 2. 加载外部验证数据 (用于测试)
    matrix_external = pd.read_csv(f'{path}/data/hGCN/expr_external_GAS_HPVCA.csv', index_col=0).T
    meta_external = pd.read_csv(f'{path}/data/hGCN/meta_external_GAS_HPVCA.csv', index_col=0)

    # 3. 标签编码
    encoder = LabelEncoder()
    # 必须用内部数据的标签拟合，保证 0/1 对应关系一致
    label_train = encoder.fit_transform(meta_train.CancerType)
    label_external = encoder.transform(meta_external.CancerType)  # Transform 外部数据

    print(f"Train Set: {matrix_train.shape}, External Set: {matrix_external.shape}")


    # --- 1. 辅助绘图样式函数 ---
    def apply_academic_axis(ax):
        ax.tick_params(axis='both', which='major', direction='out',
                       top=False, right=False, width=1.5, length=6, labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


    # --- 2. 核心逻辑函数 ---
    def train_fold(X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # 使用训练集的参数标准化外部数据

        clf = LogisticRegressionCV(
            Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
            n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
            max_iter=2000, l1_ratios=[0.5], random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        result = [np.concatenate([clf.intercept_, clf.coef_.ravel()])]
        y_test_scores = clf.predict_proba(X_test_scaled)

        onehot_encoder = OneHotEncoder(sparse_output=False)
        # 处理可能出现的单类别情况
        if len(np.unique(y_test)) < 2:
            # 如果外部验证集很小，可能缺类，这里做个简单容错
            y_test_onehot = np.zeros((len(y_test), 2))
            y_test_onehot[np.arange(len(y_test)), y_test] = 1
        else:
            y_test_onehot = onehot_encoder.fit_transform(y_test.reshape(-1, 1))

        test_aucs = []
        # 确保有两个类别的分数
        if y_test_scores.shape[1] == 2:
            try:
                # 关注 Class 1 (GAS) 的 AUC，或者多分类平均
                # 这里通常取 macro 或者针对某一类，这里逻辑保持不变
                for i in range(y_test_onehot.shape[1]):
                    auc_test = roc_auc_score(y_test_onehot[:, i], y_test_scores[:, i])
                    test_aucs.append(auc_test)
            except:
                test_aucs.append(np.nan)
        return y_test, y_test_scores, test_aucs, result


    def enhanced_evaluate_test_gene_combinations(matrix, label_train, matrix_test, label_test,
                                                 gene_combinations, gene_names,
                                                 n_repeats=1):  # External 不需要 repeat 多次，结果是固定的，设为1
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()

        results = []
        predictions_dict = {name: {'test_actual': [], 'test_scores': []} for name in gene_names}

        for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations),
                                     desc="Gene combinations"):
            if combo_idx >= len(gene_names): continue
            gene_name = gene_names[combo_idx]

            # 提取数据
            X = matrix.loc[:, genes]
            X_test_sub = matrix_test.loc[:, genes].values

            fold_metrics = {'aucs': [], 'precision_0': [], 'precision_1': [], 'recall_0': [], 'recall_1': [],
                            'f1_0': [], 'f1_1': [], 'accuracy': []}

            for _ in range(n_repeats):
                y_act, y_scr, t_aucs, _ = train_fold(X, label_train, X_test_sub, label_test)
                if len(y_scr.shape) == 2 and y_scr.shape[1] == 2: y_scr = y_scr[:, 1]
                y_act, y_scr = y_act.ravel(), y_scr.ravel()

                predictions_dict[gene_name]['test_actual'].extend(y_act)
                predictions_dict[gene_name]['test_scores'].extend(y_scr)

                y_pred = (y_scr > 0.5).astype(int)
                try:
                    p = precision_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    r = recall_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    f = f1_score(y_act, y_pred, average=None, labels=[0, 1], zero_division=0)
                    acc = accuracy_score(y_act, y_pred)
                except:
                    p, r, f = [0, 0], [0, 0], [0, 0]
                    acc = 0.0

                fold_metrics['aucs'].append(t_aucs)
                fold_metrics['accuracy'].append(acc)
                for i, m_list in enumerate([fold_metrics['precision_0'], fold_metrics['precision_1']]): m_list.append(
                    p[i])
                for i, m_list in enumerate([fold_metrics['recall_0'], fold_metrics['recall_1']]): m_list.append(r[i])
                for i, m_list in enumerate([fold_metrics['f1_0'], fold_metrics['f1_1']]): m_list.append(f[i])

            res = {'genes': genes}
            for k, v in fold_metrics.items():
                res[f'test_{k}_mean'] = np.mean(v)
                res[f'test_{k}_std'] = np.std(v)
            results.append(res)

        return pd.DataFrame(), pd.DataFrame(results), predictions_dict


    # --- 3. 绘图函数 (保持不变) ---
    def plot_roc_curves(predictions_dict, figsize=(5, 4), save_path=None, filename=None):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 14})
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)

        test_aucs = {}
        for i, (name, data) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(data['test_actual'], data['test_scores'])
            auc_val = auc(fpr, tpr)
            test_aucs[name] = auc_val
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5, alpha=0.9,
                    label=f'{name} (AUC = {auc_val:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('External Cohort ROC Curves', fontsize=18)  # 修改标题

        apply_academic_axis(ax)
        ax.legend(loc="lower right", fontsize=11, frameon=True,
                  edgecolor='black', framealpha=0.9, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_roc_curves.pdf" if filename else "roc_curves.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()
        return test_aucs


    def plot_prf_metrics(detailed_results, gene_names, class_label, dataset_type='test', save_path=None, filename=None,
                         figsize=(7, 5)):
        plt.close('all')
        metrics = ['precision', 'recall', 'f1']
        n_metrics = len(metrics)
        n_models = len(gene_names)
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.8 / n_models
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        for i, gene_name in enumerate(gene_names):
            if i >= len(detailed_results): continue
            values = []
            for metric in metrics:
                prefix = f"{dataset_type}_{metric}_{class_label}"
                values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))

            pos = np.arange(n_metrics) + i * bar_width
            bars = ax.bar(pos, values, bar_width, alpha=0.8, color=colors[i % len(colors)], label=gene_name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8)

        ax.set_xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylim(0, 1.15)

        # 动态修改标题
        if class_label == 0:
            title_text = "External Cohort PRF Metrics (Class 0, HPV-CA)"
        elif class_label == 1:
            title_text = "External Cohort PRF Metrics (Class 1, GAS)"
        else:
            title_text = f"External Cohort PRF Metrics (Class {class_label})"
        ax.set_title(title_text, fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models, fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_prf_class_{class_label}.pdf" if filename else f"prf_class_{class_label}.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_accuracy(detailed_results, gene_names, dataset_type='test', save_path=None, filename=None, figsize=(7, 5)):
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        n_models = len(gene_names)
        bar_width = 0.6
        colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

        valid_idxs = [i for i in range(len(gene_names)) if i < len(detailed_results)]
        valid_names = [gene_names[i] for i in valid_idxs]
        values = [detailed_results.iloc[i].get(f"{dataset_type}_accuracy_mean", 0) for i in valid_idxs]

        for i, (name, val) in enumerate(zip(valid_names, values)):
            ax.bar(i, val, bar_width, alpha=0.8, color=colors[i % len(colors)], label=name)
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(np.arange(len(valid_names)))
        ax.set_xticklabels(valid_names, fontsize=12)
        apply_academic_axis(ax)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
        ax.set_ylim(0, 1.15)
        ax.set_title('External Cohort Performance', fontsize=18, fontweight='normal')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(valid_names), fontsize=11,
                  frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        plt.tight_layout()
        if save_path:
            fname = f"{filename}_accuracy.pdf" if filename else "accuracy.pdf"
            plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_confusion_matrix_batch(y_true, y_pred, display_labels, title, save_path=None, filename=None):
        target_indices = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=target_indices)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        plt.close('all')
        plt.figure(figsize=(4.5, 3.5))
        plt.rcParams.update({'font.size': 14})

        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=display_labels,
                         yticklabels=display_labels,
                         annot_kws={"size": 16})

        ax.set_xticklabels(display_labels, rotation=0)
        ax.set_yticklabels(display_labels, rotation=90, va='center')

        plt.title(title, fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        if save_path:
            if not os.path.exists(save_path): os.makedirs(save_path)
            fname = filename if filename else title.replace(" ", "_")
            if not fname.endswith('.pdf'): fname += '.pdf'
            full_path = os.path.join(save_path, fname)
            plt.savefig(full_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def batch_evaluate_confusion_matrices(matrix_train, label_train, matrix_test, label_test,
                                          gene_combinations, gene_names, class_labels, save_path):
        if isinstance(label_train, torch.Tensor): label_train = label_train.cpu().numpy()
        if isinstance(label_test, torch.Tensor): label_test = label_test.cpu().numpy()
        label_train, label_test = label_train.ravel(), label_test.ravel()

        print(f"Batch CM evaluation on External Cohort...")
        for idx, (genes, name) in tqdm(enumerate(zip(gene_combinations, gene_names)), total=len(gene_names)):
            # 训练数据 (Internal)
            X_tr = matrix_train.loc[:, genes].values
            # 测试数据 (External)
            X_te = matrix_test.loc[:, genes].values

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegressionCV(Cs=[25], penalty='elasticnet', fit_intercept=False, cv=5, solver='saga',
                                       n_jobs=1, refit=True, class_weight='balanced', multi_class='ovr',
                                       max_iter=1000, l1_ratios=[0.5], random_state=42)
            try:
                clf.fit(X_tr_s, label_train)
                # y_tr_pred = clf.predict(X_tr_s) # 外部验证通常不画训练集混淆矩阵，如果需要可保留
                y_te_pred = clf.predict(X_te_s)

                # 只画外部验证集的混淆矩阵
                plot_confusion_matrix_batch(
                    y_true=label_test,
                    y_pred=y_te_pred,
                    display_labels=class_labels,
                    title=f"{name} (External Cohort)",
                    save_path=save_path,
                    filename=f"{name}_external_cm"
                )
            except Exception as e:
                print(f"Error {name}: {e}")


    # --- Execution Logic ---

    # 定义基因集 (WEDGE)
    WEDGE = ["PGC"]
    RF_gene = ["RFC3"]
    POC_gene = ["SH3BGRL"]
    Diablo_gene = ["WDR76"]
    BINN_gene = ["ILK"]

    gene_names = ["WEDGE", "BINN", 'POC-19', 'RF', 'Diablo']
    gene_comb = [WEDGE, BINN_gene, POC_gene, RF_gene, Diablo_gene]

    # 1. Evaluate (Test Set = Matrix External)
    # n_repeats = 1 因为外部验证集是固定的，不需要重复采样
    formatted_df, df_results, predictions = enhanced_evaluate_test_gene_combinations(
        matrix_train, label_train, matrix_external, label_external,
        gene_combinations=gene_comb, gene_names=gene_names, n_repeats=1
    )
    formatted_df.to_csv("H:/Proteomic/ppt/final/metric/fig/fig4/jijin/formatted_results_external.csv", index=True)
    save_fig_path = "H:/Proteomic/ppt/final/metric/fig/fig3/external/jijin/"
    os.makedirs(save_fig_path, exist_ok=True)

    # 2. Plots
    test_aucs = plot_roc_curves(predictions, save_path=save_fig_path, filename="External_Comparison")

    plot_prf_metrics(df_results, gene_names, class_label=0, dataset_type='test',
                     save_path=save_fig_path, filename="External_Comparison")
    plot_prf_metrics(df_results, gene_names, class_label=1, dataset_type='test',
                     save_path=save_fig_path, filename="External_Comparison")

    plot_accuracy(df_results, gene_names, dataset_type='test',
                  save_path=save_fig_path, filename="External_Comparison")

    # 3. Confusion Matrices
    save_dir_cm = os.path.join(save_fig_path, "cm_batch")
    display_labels = ['HPV_CA', 'GAS']

    batch_evaluate_confusion_matrices(
        matrix_train=matrix_train, label_train=label_train,
        matrix_test=matrix_external, label_test=label_external,
        gene_combinations=gene_comb, gene_names=gene_names,
        class_labels=display_labels,
        save_path=save_dir_cm
    )

    print("External Validation execution complete.")
except Exception as e:
    print(f"Error in External Validation: {e}")
    import traceback

    traceback.print_exc()
