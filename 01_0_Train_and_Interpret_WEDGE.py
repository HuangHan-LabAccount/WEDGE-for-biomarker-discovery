import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys
import os
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(os.path.join(PROJECT_ROOT, 'lib/'))
from utilsdata import *
from Train import *
from WEDGE_model import GraphLevelHeteroGCN
from WEDGE_Explain import HGCN_Node_Importance_Explainer


# Data paths - using relative paths from project root
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
STRING_DIR = os.path.join(PPI_DIR, 'String_database')
TRRUST_DIR = os.path.join(PPI_DIR, 'Trrust_database')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
GENE_RANK_DIR = os.path.join(OUTPUT_DIR, 'gene_rank')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
if True:
    # Load internal data from Demo_data
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
all_dataset = build_hetero_graph_dataset(matrix_all, adj_PPI, adj_GRN,label_all)
matrix = matrix_train
label_list = label_train
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for train_index, val_index in tqdm(skf.split(matrix, label_list), total=5, desc="Training folds"):
    fold = fold+1
    arg_label0 = pd.read_csv(os.path.join(AUG_DATA_DIR, f'generated_data_fold{fold}_0.csv'))
    arg_label0.index = arg_label0['id']
    arg_label1 = pd.read_csv(os.path.join(AUG_DATA_DIR, f'generated_data_fold{fold}_1.csv'))
    arg_label1.index = arg_label1['id']

    y_train = label_list[train_index]
    y_train = torch.tensor(np.concatenate([y_train, arg_label0['label'].values, arg_label1['label'].values]))
    y_val = label_list[val_index]

    arg_label0 = arg_label0.drop(columns=['subset', 'label', 'id'])
    arg_label1 = arg_label1.drop(columns=['subset', 'label', 'id'])
    X_train = matrix.iloc[train_index]
    X_train = pd.concat([X_train, arg_label0, arg_label1], axis=0)
    X_val = matrix.iloc[val_index]
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_val = torch.tensor(X_val.values, dtype=torch.float)

    X_test = torch.tensor(matrix_test.values, dtype=torch.float)
    y_test = torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float)

    train_dataset = build_hetero_graph_dataset(X_train, adj_PPI, adj_GRN, y_train)
    val_dataset = build_hetero_graph_dataset(X_val, adj_PPI, adj_GRN, y_val)
    test_dataset = build_hetero_graph_dataset(X_test, adj_PPI, adj_GRN, y_test)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


    trainer = create_trainer(max_epochs=2000,min_epochs = 1000, patience=100, min_delta=1e-4, log_dir="lightning_logs",
                             save_dir=CHECKPOINT_DIR, experiment_name=f"heterogcn_{fold}",
                             log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader)
    test_acc = test_results[0]['test_combined_acc']
    print(f"Fold {fold} - test_combined_acc: {test_acc:.4f}")
    if test_acc > 0.9:
        print(f"  ✓ Fold {fold} achieved >90% accuracy!")

fold = 0
for train_index, val_index in tqdm(skf.split(matrix, label_list), total=5, desc="Training folds"):
    fold = fold+1
    if fold == 5:
        y_val = label_list[val_index]
        X_val = matrix.iloc[val_index]
        X_val = torch.tensor(X_val.values, dtype=torch.float)
        X_test = torch.tensor(matrix_test.values, dtype=torch.float)
        y_test = torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float)
        val_dataset = build_hetero_graph_dataset(X_val, adj_PPI, adj_GRN, y_val)
        test_dataset = build_hetero_graph_dataset(X_test, adj_PPI, adj_GRN, y_test)

best_model = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_5/epoch=412-val_total_loss=0.3654.ckpt"))
best_model = eval(best_model.to(device))
gene_names = matrix.columns.tolist()
trainer = create_trainer(max_epochs=150, patience=1000, min_delta=1e-4, log_dir="lightning_logs",
                             save_dir="checkpoints", experiment_name=f"heterogcn_fold")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_name = 'heterogcn_5_epoch=412'
model = best_model
print(f"\nExplaining {model_name}...")


df_ave_label0_PPI, df_ave_label0_GRN, df_ave_label1_PPI, df_ave_label1_GRN = HGCN_Node_Importance_Explainer(
                model, gene_names, val_dataset, explain_type='integrated_gradients')
df_ave_label1_PPI.to_csv(os.path.join(GENE_RANK_DIR, f'{model_name}_df_ave_label1_PPI.csv'))
df_ave_label1_GRN.to_csv(os.path.join(GENE_RANK_DIR, f'{model_name}_df_ave_label1_GRN.csv'))
df_merge = pd.DataFrame({
    'ppi_score': df_ave_label1_PPI['0'],
    'grn_score': df_ave_label1_GRN['0']
})
df_merge.to_csv(os.path.join(GENE_RANK_DIR, 'merged_scores.csv'))

def scale_to_range(series, feature_range=(0, 10)):
    min_val, max_val = feature_range
    x_std = (series - series.min()) / (series.max() - series.min())
    return x_std * (max_val - min_val) + min_val
gene_rank = pd.read_csv(f'{GENE_RANK_DIR}/merged_scores.csv', index_col=0)
gene_rank['ppi_score_scaled'] = scale_to_range(gene_rank['ppi_score'])
gene_rank['grn_score_scaled'] = scale_to_range(gene_rank['grn_score'])
# gene_rank['geometric_mean'] = np.sqrt(gene_rank['ppi_score_scaled'] * gene_rank['grn_score_scaled'])
#gene_rank_merge = gene_rank['merged_rank'].sort_values(ascending=False)[0:15]
#gene_rank_subset = pd.concat([gene_rank_PPI, gene_rank_GRN, gene_rank_merge], axis=1)
gene_rank.to_csv(os.path.join(GENE_RANK_DIR, 'gene_rank_scaled.csv'))
gene_score_PPI = gene_rank['ppi_score_scaled'].sort_values(ascending=False)[0:15]
gene_score_GRN = gene_rank['grn_score_scaled'].sort_values(ascending=False)[0:15]
# gene_score_mean = gene_rank['geometric_mean'].sort_values(ascending=False)[0:20]
union_gene = pd.DataFrame(set(gene_score_PPI.index).union(set(gene_score_GRN.index)))
union_gene.to_csv(f'{GENE_RANK_DIR}/union_gene.csv')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42  # 保持矢量字体可编辑性
})
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgba

    # --- 1. 图1：PPI 分数图 (Top PPI Scores) ---
    # 假设 gene_score_PPI 已经在你的环境中定义好了 (Series类型: index=Gene, value=Score)

    if True:
        # A. 设置渐变颜色 (主色: #E64B34)
        main_color = "#E64B34"
        n_bars = len(gene_score_PPI)
        colors = []

        base_rgb = np.array(to_rgba(main_color)[:3])
        target_rgb = np.array([1.0, 1.0, 1.0])  # 白色目标

        for i in range(n_bars):
            # 渐变系数 0 -> 0.8
            factor = (i / n_bars) * 0.8
            new_color = base_rgb * (1 - factor) + target_rgb * factor
            colors.append(new_color)

        # B. 绘图初始化
        fig, ax = plt.subplots(figsize=(7.8, 3.2))

        # C. 绘制柱状图 (zorder=2, 无边框)
        bars = ax.bar(gene_score_PPI.index, gene_score_PPI.values,
                      color=colors, width=0.6, zorder=2)

        # D. 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        # E. 标题和轴标签设置
        ax.set_title('Top PPI Scores', fontsize=16, fontweight='normal', pad=15)
        ax.set_ylabel('PPI Score', fontsize=14)

        # 坐标轴刻度设置
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        # 范围限制
        ax.set_ylim(0, 11.5)
        ax.set_xlim(-0.7, len(gene_score_PPI) - 0.3)

        # F. 全包围边框 & 图层修正
        ax.grid(False)  # 关闭网格

        # 开启所有边框
        for spine_name in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_edgecolor('black')
            ax.spines[spine_name].set_linewidth(1.0)
            ax.spines[spine_name].set_zorder(10)  # 确保边框压在柱子上面

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'gene_rank/top_PPI_protein_rank_revise.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    # --- 2. 图2：GRN 分数图 (Top GRN Scores) ---
    # 假设 gene_score_GRN 已经在你的环境中定义好了

    if True:
        # A. 设置渐变颜色 (主色: #3C5487)
        main_color = "#3C5487"
        n_bars = len(gene_score_GRN)
        colors = []

        base_rgb = np.array(to_rgba(main_color)[:3])
        target_rgb = np.array([1.0, 1.0, 1.0])

        for i in range(n_bars):
            factor = (i / n_bars) * 0.8
            new_color = base_rgb * (1 - factor) + target_rgb * factor
            colors.append(new_color)

        # B. 绘图初始化
        fig, ax = plt.subplots(figsize=(7.8, 3.2))

        # C. 绘制柱状图
        bars = ax.bar(gene_score_GRN.index, gene_score_GRN.values,
                      color=colors, width=0.6, zorder=2)

        # D. 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        # E. 标题和轴标签设置
        ax.set_title('Top GRN Scores', fontsize=16, fontweight='normal', pad=15)
        ax.set_ylabel('GRN Score', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        ax.set_ylim(0, 11.5)
        ax.set_xlim(-0.7, len(gene_score_GRN) - 0.3)

        # F. 全包围边框 & 图层修正
        ax.grid(False)

        for spine_name in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_edgecolor('black')
            ax.spines[spine_name].set_linewidth(1.0)
            ax.spines[spine_name].set_zorder(10)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'gene_rank/top_GRN_protein_rank_revise.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.show()
except:
    print("Gradient color plotting failed, reverting to solid colors.")
