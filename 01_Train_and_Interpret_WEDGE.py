import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

sys.path.append(os.path.join(PROJECT_ROOT, 'lib/'))
from utilsdata import *
from Train import *
from WEDGE_model import GraphLevelHeteroGCN
from WEDGE_Explain import HGCN_Node_Importance_Explianer

import torch
from torch_geometric.data import DataLoader

# Data paths - using relative paths from project root
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Demo_data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
STRING_DIR = os.path.join(PPI_DIR, 'String_database')
TRRUST_DIR = os.path.join(PPI_DIR, 'Trrust_database')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
GENE_RANK_DIR = os.path.join(OUTPUT_DIR, 'gene_rank')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

if True:
    # Load external PPI/GRN database first for DEgene_selected
    protein_matrix = load_Stringdatabase(path=TRRUST_DIR,
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
for train_index, val_index in skf.split(matrix, label_list):
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


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphLevelHeteroGCN(
        c_in=1,
        c_hidden=256,
        c_out=2,
        lr=1e-3,
        weight_decay=1e-3,
        dp_rate=0.1,
        dp_rate_linear=0.1,
        warmup_steps=1000,  # 设置warmup步数
        label_smoothing=0.1  # 设置标签平滑
    )
    trainer = create_trainer(max_epochs=1000, patience=500, min_delta=1e-4, log_dir="lightning_logs",
                             save_dir=CHECKPOINT_DIR, experiment_name=f"heterogcn_{fold}")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

model1_1 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_1/epoch=147-val_total_loss=0.5591.ckpt"))
model1_2 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_1/epoch=136-val_total_loss=0.5640.ckpt"))
model1_3 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_1/epoch=135-val_total_loss=0.5676.ckpt"))

model2_1 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_2/epoch=89-val_total_loss=0.5169.ckpt"))
model2_2 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_2/epoch=76-val_total_loss=0.5175.ckpt"))
model2_3 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_2/epoch=73-val_total_loss=0.5151.ckpt"))

model3_1 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_3/epoch=117-val_total_loss=0.5093.ckpt"))
model3_2 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_3/epoch=113-val_total_loss=0.5039.ckpt"))
model3_3 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_3/epoch=111-val_total_loss=0.5085.ckpt"))

model4_1 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_4/epoch=77-val_total_loss=0.5042.ckpt"))
model4_2 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_4/epoch=78-val_total_loss=0.5040.ckpt"))
model4_3 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_4/epoch=82-val_total_loss=0.5166.ckpt"))

model5_1 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_5/epoch=110-val_total_loss=0.3218.ckpt"))
model5_2 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_5/epoch=160-val_total_loss=0.3272.ckpt"))
model5_3 = GraphLevelHeteroGCN.load_from_checkpoint(os.path.join(CHECKPOINT_DIR, "heterogcn_5/epoch=157-val_total_loss=0.3291.ckpt"))

trainer = create_trainer(max_epochs=150, patience=1000, min_delta=1e-4, log_dir="lightning_logs",
                             save_dir="checkpoints", experiment_name=f"heterogcn_fold")

trainer.test(model1_1, test_loader)
trainer.test(model1_2, test_loader)
trainer.test(model1_3, test_loader)

trainer.test(model2_1, test_loader)
trainer.test(model2_2, test_loader)
trainer.test(model2_3, test_loader)


trainer.test(model3_1, test_loader)
trainer.test(model3_2, test_loader)
trainer.test(model3_3, test_loader)

trainer.test(model4_1, test_loader)
trainer.test(model4_2, test_loader)
trainer.test(model4_3, test_loader)

trainer.test(model5_1, test_loader)
trainer.test(model5_2, test_loader)
trainer.test(model5_3, test_loader)

best_model = model4_3

gene_names = matrix.columns
all_dataset = build_hetero_graph_dataset(matrix_all, adj_PPI, adj_GRN, label_all)
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)
# trainer.test(model1, all_loader)
# trainer.test(model2, all_loader)
# trainer.test(model3, all_loader)
# trainer.test(model4, all_loader)
# trainer.test(model5, all_loader)

df_ave_label0_PPI, df_ave_label0_GRN, df_ave_label1_PPI, df_ave_label1_GRN = HGCN_Node_Importance_Explianer(best_model, gene_names, test_dataset, explain_type='integrated_gradients')



# Create output directories if they don't exist
os.makedirs(os.path.join(GENE_RANK_DIR, 'rep'), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df_ave_label0_PPI.to_csv(os.path.join(GENE_RANK_DIR, 'rep', 'df_ave_label0_PPI.csv'))
df_ave_label0_GRN.to_csv(os.path.join(GENE_RANK_DIR, 'rep', 'df_ave_label0_GRN.csv'))
df_ave_label1_PPI.to_csv(os.path.join(GENE_RANK_DIR, 'rep', 'df_ave_label1_PPI.csv'))
df_ave_label1_GRN.to_csv(os.path.join(GENE_RANK_DIR, 'rep', 'df_ave_label1_GRN.csv'))

# HPV_related
df_ave_label0_PPI = pd.read_csv(os.path.join(GENE_RANK_DIR, 'df_ave_label0_PPI.csv'), index_col=0).sort_values(by='0', ascending=False)
df_ave_label0_GRN = pd.read_csv(os.path.join(GENE_RANK_DIR, 'df_ave_label0_GRN.csv'), index_col=0).sort_values(by='0', ascending=False)
# EA_realted
df_ave_label1_PPI = pd.read_csv(os.path.join(GENE_RANK_DIR, 'df_ave_label1_PPI.csv'), index_col=0).sort_values(by='0', ascending=False)
df_ave_label1_GRN = pd.read_csv(os.path.join(GENE_RANK_DIR, 'df_ave_label1_GRN.csv'), index_col=0).sort_values(by='0', ascending=False)


