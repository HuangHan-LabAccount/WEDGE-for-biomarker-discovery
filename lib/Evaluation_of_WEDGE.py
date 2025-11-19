import os
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('lib/')
from utilsdata import *
from WEDGE_model import *
from Train import *
from WEDGE_model import *
from torch_geometric.data import DataLoader
os.chdir("H:/Proteomic/")
import torch
torch.__version__
if True:
    protein_matrix = load_Stringdatabase(path="H:/Proteomic/PPI_GRN_database/Trrust_database/",
                                         file_name="TF_filtered_human.csv")
    path = "H:\Proteomic"
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    matrix_Degene_sub = DEgene_selected(matrix)
    matrix = matrix_Degene_sub
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    encoder = LabelEncoder()
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    protein_matrix_PPI = load_Stringdatabase(path=os.path.join(path, 'PPI_GRN_database/String_database/'),
                                             file_name="human_PPI_score_Stringdatabase(700up).csv")
    protein_matrix_GRN = load_Stringdatabase(path=os.path.join(path, 'PPI_GRN_database/Trrust_database/'),
                                             file_name="TF_filtered_human.csv")
    adj_PPI = getAdjByString(protein_matrix_PPI, matrix_train, one_direction=False)
    adj_GRN = getAdjByString(protein_matrix_GRN, matrix_train, one_direction=True)
    gene_features = torch.tensor(matrix_train.values, dtype=torch.float)
    encoder = LabelEncoder()
path = "H:/Proteomic/"
files = os.listdir(path)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
arg_all = pd.DataFrame()
arg_label = np.array([])
for fold in range(1,6):
    arg_label0 = pd.read_csv(f'{path}/Aug_data/generated_data_fold{fold}_0.csv')
    arg_label0.index = arg_label0['id']
    arg_label1 = pd.read_csv(f'{path}/Aug_data/generated_data_fold{fold}_1.csv')
    arg_label1.index = arg_label1['id']

    arg_label = np.concatenate((arg_label,arg_label0['label'].values, arg_label1['label'].values))
    arg_label0 = arg_label0.drop(columns=['subset', 'label', 'id'])
    arg_label1 = arg_label1.drop(columns=['subset', 'label', 'id'])
    arg_all = pd.concat([arg_all, arg_label0, arg_label1], axis=0)
arg_label = torch.tensor(arg_label, dtype=torch.float)


matrix_trian = arg_all
label_list_train = arg_label

matrix_val = matrix_train
label_list_val = label_train
fold = 0

X_train = torch.tensor(matrix_trian.values, dtype=torch.float)
X_val = torch.tensor(matrix_val.values, dtype=torch.float)
y_train = label_list_train
y_val = label_list_val
X_test = torch.tensor(matrix_test.values, dtype=torch.float)
y_test = torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float)


train_dataset = build_hetero_graph_dataset(X_train, adj_PPI, adj_GRN, y_train)
val_dataset = build_hetero_graph_dataset(X_val, adj_PPI, adj_GRN, y_val)
test_dataset = build_hetero_graph_dataset(X_test, adj_PPI, adj_GRN, y_test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphLevelHeteroGCN(c_in=1,
                            c_hidden=100,
                            c_out=2,
                            lr=1e-3,
                            weight_decay=1e-3,
                            dp_rate=0.1,
                            dp_rate_linear=0.1,
                            warmup_steps=1000)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
trainer = create_trainer(max_epochs=150, patience=1000, min_delta=1e-4, log_dir="lightning_logs_all",save_dir="checkpoints_all", experiment_name=f"heterogcn_{fold}")
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)