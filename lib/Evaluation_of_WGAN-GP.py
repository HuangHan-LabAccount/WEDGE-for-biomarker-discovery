from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('lib/')
from utilsdata import *
from WEDGE_model import *
from Train import *
from WEDGE_Explain import *
from torch_geometric.data import DataLoader
if True:
    protein_matrix = load_Stringdatabase(path="H:/Proteomic/PPI_data/Trrust_database/",
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
    matrix_all = matrix_sub
    label_all = (torch.tensor(encoder.fit_transform(meta_sub.CancerType.values), dtype=torch.float))
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    protein_matrix_PPI = load_Stringdatabase(path=os.path.join(path, 'PPI_data/String_database/'),
                                             file_name="human_PPI_score_Stringdatabase(700up).csv")
    protein_matrix_GRN = load_Stringdatabase(path=os.path.join(path, 'PPI_data/Trrust_database/'),
                                             file_name="TF_filtered_human.csv")
    adj_PPI = getAdjByString(protein_matrix_PPI, matrix_train, one_direction=False)
    adj_GRN = getAdjByString(protein_matrix_GRN, matrix_train, one_direction=True)
    gene_features = torch.tensor(matrix_train.values, dtype=torch.float)
    encoder = LabelEncoder()
all_dataset = build_hetero_graph_dataset(matrix_all, adj_PPI, adj_GRN,label_all)
models_paths = {
    1: [
        "checkpoints/heterogcn_1/epoch=135-val_total_loss=0.5676.ckpt",
        "checkpoints/heterogcn_1/epoch=109-val_total_loss=0.5655.ckpt",
        "checkpoints/heterogcn_1/epoch=80-val_total_loss=0.5653.ckpt",
    ],
    2: [
        "checkpoints/heterogcn_2/epoch=73-val_total_loss=0.5151.ckpt",
        "checkpoints/heterogcn_2/epoch=72-val_total_loss=0.5166.ckpt",
        "checkpoints/heterogcn_2/epoch=71-val_total_loss=0.5058.ckpt",
    ],
    3: [
        "checkpoints/heterogcn_3/epoch=117-val_total_loss=0.5093.ckpt",
        "checkpoints/heterogcn_3/epoch=84-val_total_loss=0.5074.ckpt",
        "checkpoints/heterogcn_3/epoch=74-val_total_loss=0.5048.ckpt",
    ],
    4: [
        "checkpoints/heterogcn_4/epoch=77-val_total_loss=0.5042.ckpt",
        "checkpoints/heterogcn_4/epoch=82-val_total_loss=0.5166.ckpt",
        "checkpoints/heterogcn_4/epoch=117-val_total_loss=0.5222.ckpt",
    ],
    5: [
        "checkpoints/heterogcn_5/epoch=157-val_total_loss=0.3291.ckpt",
        "checkpoints/heterogcn_5/epoch=121-val_total_loss=0.3293.ckpt",
        "checkpoints/heterogcn_5/epoch=162-val_total_loss=0.3296.ckpt",
    ]
}


def evaluate_model_with_trainer(trainer, model, test_loader):
    test_results = trainer.test(model, test_loader)
    test_protein_acc = test_results[0]['test_protein_acc']
    test_gene_acc = test_results[0]['test_gene_acc']
    test_combined_acc = test_results[0]['test_combined_acc']

    return {
        'test_protein_acc': test_protein_acc,
        'test_gene_acc': test_gene_acc,
        'test_combined_acc': test_combined_acc
    }


results = []
trainer = create_trainer(max_epochs=150, patience=1000, min_delta=1e-4, log_dir="lightning_logs",
                             save_dir="checkpoints", experiment_name=f"heterogcn_fold")
# 设置参数
path = "H:/Proteomic/"
files = os.listdir(path)
matrix = matrix_train
label_list = label_train
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("开始评估所有折和所有模型...")
print("=" * 80)
fold = 0
for train_index, val_index in skf.split(matrix, label_list):
    fold += 1
    print(f"\n处理第 {fold} 折...")
    arg_label0 = pd.read_csv(f'{path}/Aug_data/generated_data_fold{fold}_0.csv')
    arg_label0.index = arg_label0['id']
    arg_label1 = pd.read_csv(f'{path}/Aug_data/generated_data_fold{fold}_1.csv')
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

    from torch_geometric.loader import DataLoader as GeometricDataLoader

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    fold_models = models_paths[fold]
    for model_idx, model_path in enumerate(fold_models, 1):
        print(f"  evaluate model {model_idx}: {model_path}")

        try:
            model = GraphLevelHeteroGCN.load_from_checkpoint(model_path)
            metrics = evaluate_model_with_trainer(trainer, model, test_loader)
            result = {
                'Fold': fold,
                'Model': model_idx,
                'Model_Path': model_path,
                'Test_Protein_Acc': metrics['test_protein_acc'],
                'Test_Gene_Acc': metrics['test_gene_acc'],
                'Test_Combined_Acc': metrics['test_combined_acc']
            }
            results.append(result)

            print(
                f"    test cohort - Protein: {metrics['test_protein_acc']:.4f}, Gene: {metrics['test_gene_acc']:.4f}, Combined: {metrics['test_combined_acc']:.4f}")

        except Exception as e:
            print(f"    failed: {e}")
            result = {
                'Fold': fold,
                'Model': model_idx,
                'Model_Path': model_path,
                'Test_Protein_Acc': None,
                'Test_Gene_Acc': None,
                'Test_Combined_Acc': None
            }
            results.append(result)

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("resluts:")
print("=" * 80)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("summary:")
print("=" * 80)