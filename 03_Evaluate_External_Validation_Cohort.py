import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

sys.path.append(os.path.join(PROJECT_ROOT, 'lib/'))
from utilsdata import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data paths - using relative paths from project root
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Demo_data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]
# RandomForest Merics
RF_gene = ["CDK4", "MCMBP"]
POC_gene = ["SH3BGRL", "FMO1"]
WEDGE = ["PGC", "DNMT1"]
Diablo_gene = ["WDR76", "MCMBP"]
BINN_gene = ["ILK", "ACTB"]


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
        penalty='elasticnet',  # 使用 ElasticNet 正则化
        fit_intercept=False,
        cv=5,
        solver='saga',  # 使用 saga solver，支持 ElasticNet
        n_jobs=1,
        refit=True,
        class_weight='balanced',
        multi_class='ovr',
        max_iter=1000,
        l1_ratios=[0.5]
    )
    clf.fit(X_train_scaled, y_train)

    coef = clf.coef_.ravel()
    result = [np.concatenate([clf.intercept_, coef])]

    y_val_scores = clf.predict_proba(X_val_scaled)
    y_test_scores = clf.predict_proba(X_test_scaled)

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
            print(f"AUC calculation error for class {i}: Possible insufficient samples or imbalanced labels")
            val_aucs.append(np.nan)
            test_aucs.append(np.nan)
        except Exception as e:
            print(f"Unknown error occurred for class {i}: {str(e)}")
            val_aucs.append(np.nan)
            test_aucs.append(np.nan)

    return y_val, y_val_scores, y_test, y_test_scores, val_aucs, test_aucs, result


def evaluate_test_gene_combinations(matrix, label, gene_combinations, gene_names, matrix_test, label_test,
                                    n_splits=5, n_repeats=1):
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(label_test, torch.Tensor):
        label_test = label_test.cpu().numpy()

    results = []
    y = label
    y_test = label_test

    for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations), desc="Gene combinations"):
        X = matrix.loc[:, genes]
        X_test = matrix_test.loc[:, genes].values
        fold_val_metrics = {
            'aucs': [], 'precision_0': [], 'precision_1': [],
            'recall_0': [], 'recall_1': [],
            'f1_0': [], 'f1_1': [], 'accuracy': []
        }
        fold_test_metrics = {
            'aucs': [], 'precision_0': [], 'precision_1': [],
            'recall_0': [], 'recall_1': [],
            'f1_0': [], 'f1_1': [], 'accuracy': []
        }

        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train = X.iloc[train_idx].values
                X_val = X.iloc[val_idx].values

                y_train, y_val = y[train_idx], y[val_idx]

                y_val_actual, y_val_scores, y_test_actual, y_test_scores, val_aucs, test_aucs, model = train_fold(
                    X_train, y_train, X_val, y_val, X_test, y_test)

                if isinstance(y_val_scores, torch.Tensor):
                    y_val_scores = y_val_scores.cpu().numpy()
                if isinstance(y_test_scores, torch.Tensor):
                    y_test_scores = y_test_scores.cpu().numpy()

                if len(y_val_scores.shape) == 2 and y_val_scores.shape[1] == 2:
                    y_val_scores = y_val_scores[:, 1]
                if len(y_test_scores.shape) == 2 and y_test_scores.shape[1] == 2:
                    y_test_scores = y_test_scores[:, 1]

                if isinstance(y_val_actual, torch.Tensor):
                    y_val_actual = y_val_actual.cpu().numpy()
                if isinstance(y_test_actual, torch.Tensor):
                    y_test_actual = y_test_actual.cpu().numpy()

                y_val_actual = y_val_actual.ravel()
                y_test_actual = y_test_actual.ravel()
                y_val_scores = y_val_scores.ravel()
                y_test_scores = y_test_scores.ravel()

                y_val_pred = (y_val_scores > 0.5).astype(int)
                y_test_pred = (y_test_scores > 0.5).astype(int)

                try:
                    val_precision = precision_score(y_val_actual, y_val_pred, average=None, labels=[0, 1],
                                                    zero_division=0)
                    val_recall = recall_score(y_val_actual, y_val_pred, average=None, labels=[0, 1], zero_division=0)
                    val_f1 = f1_score(y_val_actual, y_val_pred, average=None, labels=[0, 1], zero_division=0)
                    val_accuracy = accuracy_score(y_val_actual, y_val_pred)
                except Exception as e:
                    val_precision = np.array([0.0, 0.0])
                    val_recall = np.array([0.0, 0.0])
                    val_f1 = np.array([0.0, 0.0])
                    val_accuracy = 0.0

                try:
                    test_precision = precision_score(y_test_actual, y_test_pred, average=None, labels=[0, 1],
                                                     zero_division=0)
                    test_recall = recall_score(y_test_actual, y_test_pred, average=None, labels=[0, 1], zero_division=0)
                    test_f1 = f1_score(y_test_actual, y_test_pred, average=None, labels=[0, 1], zero_division=0)
                    test_accuracy = accuracy_score(y_test_actual, y_test_pred)
                except Exception as e:
                    test_precision = np.array([0.0, 0.0])
                    test_recall = np.array([0.0, 0.0])
                    test_f1 = np.array([0.0, 0.0])
                    test_accuracy = 0.0

                fold_val_metrics['aucs'].append(val_aucs)
                fold_val_metrics['precision_0'].append(val_precision[0])
                fold_val_metrics['precision_1'].append(val_precision[1])
                fold_val_metrics['recall_0'].append(val_recall[0])
                fold_val_metrics['recall_1'].append(val_recall[1])
                fold_val_metrics['f1_0'].append(val_f1[0])
                fold_val_metrics['f1_1'].append(val_f1[1])
                fold_val_metrics['accuracy'].append(val_accuracy)

                fold_test_metrics['aucs'].append(test_aucs)
                fold_test_metrics['precision_0'].append(test_precision[0])
                fold_test_metrics['precision_1'].append(test_precision[1])
                fold_test_metrics['recall_0'].append(test_recall[0])
                fold_test_metrics['recall_1'].append(test_recall[1])
                fold_test_metrics['f1_0'].append(test_f1[0])
                fold_test_metrics['f1_1'].append(test_f1[1])
                fold_test_metrics['accuracy'].append(test_accuracy)

        result = {'genes': genes}

        for metric in fold_val_metrics:
            values = np.array(fold_val_metrics[metric])
            result[f'val_{metric}_mean'] = np.mean(values)
            result[f'val_{metric}_std'] = np.std(values)

        for metric in fold_test_metrics:
            values = np.array(fold_test_metrics[metric])
            result[f'test_{metric}_mean'] = np.mean(values)
            result[f'test_{metric}_std'] = np.std(values)

        results.append(result)

    df_results = pd.DataFrame(results)

    formatted_data = []
    for i, gene_name in enumerate(gene_names):
        if i < len(df_results):
            row_0 = {
                'geneset_name': gene_name,
                'class': 0,
                'precision': df_results.iloc[i]['test_precision_0_mean'],
                'recall': df_results.iloc[i]['test_recall_0_mean'],
                'f1': df_results.iloc[i]['test_f1_0_mean'],
                'accuracy': df_results.iloc[i]['test_accuracy_mean']
            }
            row_1 = {
                'geneset_name': gene_name,
                'class': 1,
                'precision': df_results.iloc[i]['test_precision_1_mean'],
                'recall': df_results.iloc[i]['test_recall_1_mean'],
                'f1': df_results.iloc[i]['test_f1_1_mean'],
                'accuracy': df_results.iloc[i]['test_accuracy_mean']
            }
            formatted_data.append(row_0)
            formatted_data.append(row_1)

    formatted_df = pd.DataFrame(formatted_data)
    return formatted_df, df_results


def enhanced_evaluate_test_gene_combinations(matrix, label, gene_combinations, gene_names, matrix_test, label_test,
                                             n_splits=5, n_repeats=1):
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(label_test, torch.Tensor):
        label_test = label_test.cpu().numpy()

    results = []
    y = label
    y_test = label_test

    predictions_dict = {}
    for gene_name in gene_names:
        predictions_dict[gene_name] = {
            'val_actual': [], 'val_scores': [], 'test_actual': [], 'test_scores': []
        }

    for combo_idx, genes in tqdm(enumerate(gene_combinations), total=len(gene_combinations), desc="Gene combinations"):
        if combo_idx >= len(gene_names):
            continue

        gene_name = gene_names[combo_idx]
        X = matrix.loc[:, genes]
        X_test = matrix_test.loc[:, genes].values
        fold_val_metrics = {
            'aucs': [], 'precision_0': [], 'precision_1': [],
            'recall_0': [], 'recall_1': [],
            'f1_0': [], 'f1_1': [], 'accuracy': []
        }
        fold_test_metrics = {
            'aucs': [], 'precision_0': [], 'precision_1': [],
            'recall_0': [], 'recall_1': [],
            'f1_0': [], 'f1_1': [], 'accuracy': []
        }

        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train = X.iloc[train_idx].values
                X_val = X.iloc[val_idx].values
                y_train, y_val = y[train_idx], y[val_idx]

                y_val_actual, y_val_scores, y_test_actual, y_test_scores, val_aucs, test_aucs, model = train_fold(
                    X_train, y_train, X_val, y_val, X_test, y_test)

                if isinstance(y_val_scores, torch.Tensor):
                    y_val_scores = y_val_scores.cpu().numpy()
                if isinstance(y_test_scores, torch.Tensor):
                    y_test_scores = y_test_scores.cpu().numpy()

                if len(y_val_scores.shape) == 2 and y_val_scores.shape[1] == 2:
                    y_val_scores = y_val_scores[:, 1]
                if len(y_test_scores.shape) == 2 and y_test_scores.shape[1] == 2:
                    y_test_scores = y_test_scores[:, 1]

                if isinstance(y_val_actual, torch.Tensor):
                    y_val_actual = y_val_actual.cpu().numpy()
                if isinstance(y_test_actual, torch.Tensor):
                    y_test_actual = y_test_actual.cpu().numpy()

                y_val_actual = y_val_actual.ravel()
                y_test_actual = y_test_actual.ravel()
                y_val_scores = y_val_scores.ravel()
                y_test_scores = y_test_scores.ravel()

                predictions_dict[gene_name]['val_actual'].extend(y_val_actual)
                predictions_dict[gene_name]['val_scores'].extend(y_val_scores)
                predictions_dict[gene_name]['test_actual'].extend(y_test_actual)
                predictions_dict[gene_name]['test_scores'].extend(y_test_scores)

                y_val_pred = (y_val_scores > 0.5).astype(int)
                y_test_pred = (y_test_scores > 0.5).astype(int)

                try:
                    val_precision = precision_score(y_val_actual, y_val_pred, average=None, labels=[0, 1],
                                                    zero_division=0)
                    val_recall = recall_score(y_val_actual, y_val_pred, average=None, labels=[0, 1], zero_division=0)
                    val_f1 = f1_score(y_val_actual, y_val_pred, average=None, labels=[0, 1], zero_division=0)
                    val_accuracy = accuracy_score(y_val_actual, y_val_pred)
                except Exception as e:
                    val_precision = np.array([0.0, 0.0])
                    val_recall = np.array([0.0, 0.0])
                    val_f1 = np.array([0.0, 0.0])
                    val_accuracy = 0.0

                try:
                    test_precision = precision_score(y_test_actual, y_test_pred, average=None, labels=[0, 1],
                                                     zero_division=0)
                    test_recall = recall_score(y_test_actual, y_test_pred, average=None, labels=[0, 1], zero_division=0)
                    test_f1 = f1_score(y_test_actual, y_test_pred, average=None, labels=[0, 1], zero_division=0)
                    test_accuracy = accuracy_score(y_test_actual, y_test_pred)
                except Exception as e:
                    test_precision = np.array([0.0, 0.0])
                    test_recall = np.array([0.0, 0.0])
                    test_f1 = np.array([0.0, 0.0])
                    test_accuracy = 0.0

                fold_val_metrics['aucs'].append(val_aucs)
                fold_val_metrics['precision_0'].append(val_precision[0])
                fold_val_metrics['precision_1'].append(val_precision[1])
                fold_val_metrics['recall_0'].append(val_recall[0])
                fold_val_metrics['recall_1'].append(val_recall[1])
                fold_val_metrics['f1_0'].append(val_f1[0])
                fold_val_metrics['f1_1'].append(val_f1[1])
                fold_val_metrics['accuracy'].append(val_accuracy)

                fold_test_metrics['aucs'].append(test_aucs)
                fold_test_metrics['precision_0'].append(test_precision[0])
                fold_test_metrics['precision_1'].append(test_precision[1])
                fold_test_metrics['recall_0'].append(test_recall[0])
                fold_test_metrics['recall_1'].append(test_recall[1])
                fold_test_metrics['f1_0'].append(test_f1[0])
                fold_test_metrics['f1_1'].append(test_f1[1])
                fold_test_metrics['accuracy'].append(test_accuracy)

        result = {'genes': genes}

        for metric in fold_val_metrics:
            values = np.array(fold_val_metrics[metric])
            result[f'val_{metric}_mean'] = np.mean(values)
            result[f'val_{metric}_std'] = np.std(values)

        for metric in fold_test_metrics:
            values = np.array(fold_test_metrics[metric])
            result[f'test_{metric}_mean'] = np.mean(values)
            result[f'test_{metric}_std'] = np.std(values)

        results.append(result)

    df_results = pd.DataFrame(results)

    formatted_data = []
    for i, gene_name in enumerate(gene_names):
        if i < len(df_results):
            row_0 = {
                'geneset_name': gene_name, 'class': 0,
                'precision': df_results.iloc[i]['test_precision_0_mean'],
                'recall': df_results.iloc[i]['test_recall_0_mean'],
                'f1': df_results.iloc[i]['test_f1_0_mean'],
                'accuracy': df_results.iloc[i]['test_accuracy_mean']
            }
            row_1 = {
                'geneset_name': gene_name, 'class': 1,
                'precision': df_results.iloc[i]['test_precision_1_mean'],
                'recall': df_results.iloc[i]['test_recall_1_mean'],
                'f1': df_results.iloc[i]['test_f1_1_mean'],
                'accuracy': df_results.iloc[i]['test_accuracy_mean']
            }
            formatted_data.append(row_0)
            formatted_data.append(row_1)

    formatted_df = pd.DataFrame(formatted_data)
    return formatted_df, df_results, predictions_dict


def plot_roc_curves(predictions_dict, figsize=(8, 4), save_path=None, filename=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.close('all')  # 关闭所有之前的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})

    colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

    # 配置第一个子图 (验证集)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=16)
    ax1.set_ylabel('True Positive Rate', fontsize=16)
    ax1.set_title('Validation ROC Curves', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(False)  # 关闭网格

    # 配置第二个子图 (测试集)
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=16)
    ax2.set_ylabel('True Positive Rate', fontsize=16)
    ax2.set_title('Testing ROC Curves', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(False)  # 关闭网格

    val_aucs = {}
    test_aucs = {}

    for i, (gene_name, pred_data) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]

        val_fpr, val_tpr, _ = roc_curve(pred_data['val_actual'], pred_data['val_scores'])
        val_auc = auc(val_fpr, val_tpr)
        val_aucs[gene_name] = val_auc

        ax1.plot(val_fpr, val_tpr, color=color,
                 label=f'{gene_name} (AUC = {val_auc:.2f})', lw=2, alpha=0.8)

        test_fpr, test_tpr, _ = roc_curve(pred_data['test_actual'], pred_data['test_scores'])
        test_auc = auc(test_fpr, test_tpr)
        test_aucs[gene_name] = test_auc

        ax2.plot(test_fpr, test_tpr, color=color,
                 label=f'{gene_name} (AUC = {test_auc:.2f})', lw=2, alpha=0.8)

    ax1.legend(loc="lower right", fontsize=10, frameon=True, handlelength=1.5,
               handletextpad=0.5, framealpha=0.8, edgecolor='lightgray')
    ax2.legend(loc="lower right", fontsize=10, frameon=True, handlelength=1.5,
               handletextpad=0.5, framealpha=0.8, edgecolor='lightgray')

    plt.tight_layout()

    if save_path:
        if filename:
            plt.savefig(f"{save_path}/{filename}_roc_curves.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/roc_curves.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return val_aucs, test_aucs


def plot_prf_metrics(detailed_results, gene_names, class_label, dataset_type='test', save_path=None, filename=None,
                     figsize=(7, 4)):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')  # 关闭所有之前的图形

    metrics = ['precision', 'recall', 'f1']
    n_metrics = len(metrics)
    n_models = len(gene_names)

    plt.figure(figsize=figsize)

    plt.rcParams.update({'font.size': 14})

    bar_width = 0.8 / n_models
    colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

    for i, gene_name in enumerate(gene_names):
        if i >= len(detailed_results): continue

        values, errors = [], []
        for metric in metrics:
            prefix = f"{dataset_type}_{metric}_{class_label}"
            values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))
            errors.append(detailed_results.iloc[i].get(f"{prefix}_std", 0))

        pos = np.arange(n_metrics) + i * bar_width
        bars = plt.bar(pos, values, bar_width, alpha=0.8, color=colors[i % len(colors)],
                       label=gene_name,
                       # yerr=errors,
                       capsize=3, ecolor='black',
                       # error_kw={'elinewidth': 1}
                       )

        for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
            label_height = val + err + 0.02
            plt.text(bar.get_x() + bar.get_width() / 2, label_height,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2, metrics, fontsize=12)
    plt.yticks(fontsize=12)
    dataset_name = "Inner Val Dataset" if dataset_type == 'val' else "Outer Test Dataset"
    plt.title(f'Class {class_label} in {dataset_name} PRF Metrics', fontsize=12)
    plt.ylim(0, 1.1)
    plt.yticks([])  # 不显示y轴坐标
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models, fontsize=12)  # 图例横着铺开在底部
    plt.tight_layout()
    plt.grid(False)  # 关闭网格

    if save_path:
        if filename:
            plt.savefig(f"{save_path}/{filename}_prf_metrics_class_{class_label}_{dataset_type}.pdf", dpi=300,
                        bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/prf_metrics_class_{class_label}_{dataset_type}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_accuracy(detailed_results, gene_names, dataset_type='test', save_path=None, filename=None, figsize=(7, 4)):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')

    plt.figure(figsize=figsize)

    plt.rcParams.update({'font.size': 14})

    n_models = len(gene_names)
    bar_width = 0.6  # 较宽的条形
    colors = ["#E64B34", "#3C5487", "#00A088", "#F4AE64", "#4DBBD6", "#785390", "#F29B80", "#8DD3C6"]

    values, errors = [], []
    valid_gene_names = []  # Store gene names that have valid data

    for i, gene_name in enumerate(gene_names):
        if i >= len(detailed_results): continue

        prefix = f"{dataset_type}_accuracy"
        values.append(detailed_results.iloc[i].get(f"{prefix}_mean", 0))
        errors.append(detailed_results.iloc[i].get(f"{prefix}_std", 0))
        valid_gene_names.append(gene_name)

    # Create a bar for each gene with its appropriate label
    for i, (gene_name, val, err) in enumerate(zip(valid_gene_names, values, errors)):
        # Add the label parameter here
        bar = plt.bar(i, val, bar_width, alpha=0.8,
                      color=colors[i % len(colors)],
                      # yerr=err,
                      capsize=3,
                      ecolor='black',
                      # error_kw={'elinewidth': 1},
                      label=gene_name)

        # 在误差棒顶部显示数值
        label_height = val + err + 0.02  # 在误差棒顶部上方添加标签
        plt.text(i, label_height, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(np.arange(len(valid_gene_names)), valid_gene_names, fontsize=12)
    plt.yticks(fontsize=12)
    dataset_name = "Inner Val Dataset" if dataset_type == 'val' else "Outer Test Dataset"
    plt.title(f'Diff Method in {dataset_type} Accuracy', fontsize=18)
    plt.ylim(0, 1.1)
    plt.yticks([])  # 不显示y轴坐标
    plt.ylabel('Accuracy', fontsize=16)
    plt.grid(False)  # 关闭网格

    # Force legend to be in a single row by setting ncol to the number of models
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(valid_gene_names), fontsize=12)

    # Add more bottom margin to accommodate the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Adjust this value as needed based on the number of models

    if save_path:
        if filename:
            plt.savefig(f"{save_path}/{filename}_accuracy_{dataset_type}.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/accuracy_{dataset_type}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()


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


def evaluate_model_CFM(model, X_train, X_test, y_train, y_test, model_name="Model", title='title', save_path=None,
                       filename=None,
                       **kwargs):
    # 获取预测标签（用于混淆矩阵和准确率）
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算准确率
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_cm = plot_confusion_matrix(y_train, y_train_pred,
                                     title=f"{title}",
                                     save_path=save_path, filename=f"{filename}_train_cm" if filename else None)
    test_cm = plot_confusion_matrix(y_test, y_test_pred,
                                    title=f"{title}",
                                    save_path=save_path, filename=f"{filename}_test_cm" if filename else None)

    print(f"\n{model_name} - Performance Metrics:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm
    }


if True:
    # Load external PPI/GRN database first for DEgene_selected
    protein_matrix = load_Stringdatabase(path=os.path.join(PPI_DIR, 'Trrust_database'),
                                         file_name="TF_filtered_human.csv")

    # Load internal data from Demo_data
    meta = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_selected.csv'), index_col=0)
    matrix = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T
    matrix_Degene_sub = DEgene_selected(matrix, deg_path=PPI_DIR)
    matrix = matrix_Degene_sub
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    encoder = LabelEncoder()
    matrix_all = matrix_sub
    label_all = (torch.tensor(encoder.fit_transform(meta_sub.CancerType.values), dtype=torch.float))
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    label_test = (torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float))
    protein_matrix_PPI = load_Stringdatabase(path=os.path.join(PPI_DIR, 'String_database'),
                                             file_name="human_PPI_score_Stringdatabase(700up).csv")
    protein_matrix_GRN = load_Stringdatabase(path=os.path.join(PPI_DIR, 'Trrust_database'),
                                             file_name="TF_filtered_human.csv")
    adj_PPI = getAdjByString(protein_matrix_PPI, matrix_train, one_direction=False)
    adj_GRN = getAdjByString(protein_matrix_GRN, matrix_train, one_direction=True)
    gene_features = torch.tensor(matrix_train.values, dtype=torch.float)
    encoder = LabelEncoder()
matrix_external= pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_external_GAS_HPVCA.csv'), index_col=0).T
meta_external = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_external_GAS_HPVCA.csv'), index_col=0)
label_external = (torch.tensor(encoder.fit_transform(meta_external.CancerType.values), dtype=torch.float))

# ==============================================================================
# 1. 设置路径与加载外部数据
# ==============================================================================
save_path = os.path.join(OUTPUT_DIR, "fig/fig3/external/")
os.makedirs(save_path, exist_ok=True)

try:
    import pandas as pd
    import numpy as np
    import os
    import torch
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    # ==============================================================================
    # 0. 准备工作：设置路径与基因
    # ==============================================================================
    save_path = os.path.join(OUTPUT_DIR, "fig/fig3/external/")
    os.makedirs(save_path, exist_ok=True)

    # 定义 WEDGE 基因组
    target_genes = ["PGC", "DNMT1"]
    gene_set_name = "WEDGE_External"  # 图表标题用

    # ==============================================================================
    # 1. 准备数据：Internal Train vs External
    # ==============================================================================
    print("1. Preparing Data...")

    # --- A. 准备内部训练集 (Internal Train) ---
    # 确保使用与最初训练时完全一致的数据
    X_train = matrix_train.loc[:, target_genes]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(meta_train.CancerType)

    # --- B. 加载并准备外部验证集 (External Cohort) ---
    # 加载数据 (注意转置以匹配 样本x基因 的格式)
    matrix_external = pd.read_csv(f'{path}/data/hGCN/expr_external_GAS_HPVCA.csv', index_col=0).T
    meta_external = pd.read_csv(f'{path}/data/hGCN/meta_external_GAS_HPVCA.csv', index_col=0)

    X_ext = matrix_external[target_genes]
    y_ext = encoder.transform(meta_external['CancerType'])  # 使用训练集的encoder

    print(f"   Train Set: {X_train.shape[0]} samples")
    print(f"   External Set: {X_ext.shape[0]} samples")

    # --- C. 标准化 ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 关键：使用训练集的均值和方差来标准化外部数据
    X_ext_scaled = scaler.transform(X_ext)

    # ==============================================================================
    # 2. 模型训练 (仅在 Internal Train 上训练)
    # ==============================================================================
    print("2. Training Model on Internal Data...")

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

    # ==============================================================================
    # 3. 预测与评估
    # ==============================================================================
    print("3. Evaluating on External Data...")

    # 获取预测结果
    y_train_scores = clf.predict_proba(X_train_scaled)[:, 1]  # 用于对比 ROC
    y_ext_scores = clf.predict_proba(X_ext_scaled)[:, 1]  # 核心关注点
    y_ext_pred = clf.predict(X_ext_scaled)
    evaluate_model_CFM(
        model=clf,
        X_train=X_train_scaled,
        X_test=X_ext_scaled,
        y_train=y_train,
        y_test=y_ext,
        model_name=f"{gene_set_name}",
        title=f"External Validation Cohort",
        save_path=save_path,
        filename=f"{gene_set_name}"
    )

    # --- B. ROC 曲线 ---
    # 构造数据字典：对比 Training 效果 vs External 效果
    # (注意：如果想对比 Internal Test vs External，可以将 val_actual/scores 替换为 y_test/y_test_scores)
    predictions_dict = {
        gene_set_name: {
            'val_actual': y_train,
            'val_scores': y_train_scores,
            'test_actual': y_ext,
            'test_scores': y_ext_scores
        }
    }

    plot_roc_curves(
        predictions_dict,
        save_path=save_path,
        filename=f"{gene_set_name}"
    )

    # --- C. PRF 和 Accuracy ---
    # 计算指标
    precision = precision_score(y_ext, y_ext_pred, average=None, labels=[0, 1], zero_division=0)
    recall = recall_score(y_ext, y_ext_pred, average=None, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_ext, y_ext_pred, average=None, labels=[0, 1], zero_division=0)
    accuracy = accuracy_score(y_ext, y_ext_pred)

    # 构造 DataFrame
    ext_results = [{
        'genes': target_genes,
        # Class 0
        'test_precision_0_mean': precision[0], 'test_precision_0_std': 0,
        'test_recall_0_mean': recall[0], 'test_recall_0_std': 0,
        'test_f1_0_mean': f1[0], 'test_f1_0_std': 0,
        # Class 1
        'test_precision_1_mean': precision[1], 'test_precision_1_std': 0,
        'test_recall_1_mean': recall[1], 'test_recall_1_std': 0,
        'test_f1_1_mean': f1[1], 'test_f1_1_std': 0,
        # Accuracy
        'test_accuracy_mean': accuracy, 'test_accuracy_std': 0
    }]
    df_ext = pd.DataFrame(ext_results)

    # 绘图
    # 注意：dataset_type='test' 在这里代指 External Set
    plot_prf_metrics(df_ext, [gene_set_name], class_label=0, dataset_type='test',
                     save_path=save_path, filename=f"External Validation Cohort", figsize=(4, 4))
    plot_prf_metrics(df_ext, [gene_set_name], class_label=1, dataset_type='test',
                     save_path=save_path, filename=f"External Validation Cohort", figsize=(4, 4))
    plot_accuracy(df_ext, [gene_set_name], dataset_type='test',
                  save_path=save_path, filename=f"External Validation Cohort", figsize=(4, 4))

    print("Done. Check results in:", save_path)
except:
    print("Something went wrong!")