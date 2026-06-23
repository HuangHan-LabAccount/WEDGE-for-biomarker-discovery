# https://github.com/InfectionMedicineProteomics/BINN
# https://pmc.ncbi.nlm.nih.gov/articles/PMC10475049/
# Uniport
from binn import BINN, BINNDataLoader, BINNTrainer, BINNExplainer
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import LabelEncoder
import torch
import sys
import numpy as np

sys.path.append('../lib/')
from utilsdata import *
if True:
    path = "H:\Proteomic"
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    df = matrix.columns
    matrix_Degene_sub = DEgene_selected(matrix)
    matrix = matrix_Degene_sub
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    encoder = LabelEncoder()
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    encoder = LabelEncoder()
    label_test = (torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float))
    X_train = matrix_train
    y_train = label_train
    X_test = matrix_test
    y_test = label_test
y_train_df = pd.DataFrame({
    'sample': X_train.index,
    'group': y_train.numpy()
})


y_test_df = pd.DataFrame({
    'sample': X_test.index,
    'group': y_test.numpy()
})
uniport_mapping = pd.read_table('H:/Proteomic/compare_model/BINN/idmapping_2025_04_11.tsv')

from binn.model.util import load_reactome_db
reactome_db = load_reactome_db(input_source="uniprot")
reactome_mapping = reactome_db["mapping"]
common_entries = np.isin(uniport_mapping['Entry'].values, reactome_mapping['input'].values)
common_entries_unique = uniport_mapping[common_entries].drop_duplicates(subset=['From'])





common_entries_2 = uniport_mapping[common_entries]
one_entry_per_from = common_entries_2.drop_duplicates(subset=['From'])
valid_proteins = one_entry_per_from[one_entry_per_from['From'].isin(X_train.columns)]
protein_to_uniprot = dict(zip(valid_proteins['From'], valid_proteins['Entry']))
X_train_filtered = X_train[valid_proteins['From'].values]
X_train_filtered.columns = [protein_to_uniprot[col] for col in X_train_filtered.columns]
X_train_transpose = X_train_filtered.T
X_train_transpose['Protein'] = X_train_transpose.index
print(f"X_train_transpose unique index: {X_train_transpose.index.is_unique}")
if not X_train_transpose.index.is_unique:
    duplicated_indices = X_train_transpose.index[X_train_transpose.index.duplicated()].tolist()
    print(f"duplicated indices: {duplicated_indices}")
    X_train_transpose = X_train_transpose.loc[~X_train_transpose.index.duplicated(keep='first')]

X_test_filtered = X_test[valid_proteins['From'].values]
X_test_filtered.columns = [protein_to_uniprot[col] for col in X_test_filtered.columns]
X_test_transpose = X_test_filtered.T
X_test_transpose['Protein'] = X_test_transpose.index
if not X_test_transpose.index.is_unique:
    duplicated_indices = X_test_transpose.index[X_test_transpose.index.duplicated()].tolist()
    print(f"duplicated indices: {duplicated_indices}")
    X_test_transpose = X_test_transpose.loc[~X_test_transpose.index.duplicated(keep='first')]

data_matrix = pd.DataFrame(index=X_train_transpose.index)
data_matrix.reset_index(inplace=True)
data_matrix.rename(columns={'index': 'Protein'}, inplace=True)
y_train_df = pd.DataFrame({
    'sample': X_train.index,
    'group': y_train.numpy()
})


binn = BINN(
    data_matrix=data_matrix,
    network_source="reactome",
    n_layers=4,
    dropout=0.2
)
binn_dataloader = BINNDataLoader(binn)

dataloaders = binn_dataloader.create_dataloaders(
    data_matrix=X_train_transpose,
    design_matrix=y_train_df,
    feature_column="Protein",
    group_column="group",
    sample_column="sample",
    batch_size=32,
    validation_split=0.2,
)

dataloaders_test = binn_dataloader.create_dataloaders(
    data_matrix=X_test_transpose,
    design_matrix=y_test_df,
    feature_column="Protein",
    group_column="group",
    sample_column="sample",
    batch_size=32,
    validation_split=0.2,
)

trainer = BINNTrainer(binn)
trainer.fit(dataloaders=dataloaders, num_epochs=100)
test_results = trainer.evaluate(dataloaders_test['train'])
explainer = BINNExplainer(binn)
explanations = explainer.explain_single(
    dataloaders_test
)
from binn.plot.network import visualize_binn

layer_specific_top_n = {"0": 10, "1": 7, "2": 5, "3": 5, "4": 5}
plt = visualize_binn(
    explanations,
    top_n=layer_specific_top_n,
    plot_size=(20, 15),
    sink_node_size=500,
    node_size_scaling=200,
    edge_width=1,
    node_cmap="coolwarm",
    pathways_mapping="reactome",
    input_entity_mapping="uniprot"
)
plt.title("Interpreted network")
plt.savefig("H:/Proteomic/compare_model/BINN/interpreted_binn.png")
average_explanations = explainer.explain(
    dataloaders, nr_iterations=3, num_epochs=50, trainer=trainer
)
normalized_average_explanations = explainer.normalize_importances(
    average_explanations, method="fan")

average_importances = average_explanations
average_importances["copy"] = average_importances.apply(lambda x: True if x["source_node"] == x["target_node"] else False, axis=1)
average_importances = average_importances[average_importances["copy"] == False]
importance_df_copy = (
    average_importances
    .groupby(["source_node", "source_layer", "target_layer"], as_index=False)
    .mean(numeric_only=True)
)

def compute_normalized_ranks(df, num_importance=5):
    for i in range(num_importance):
        col = f"importance_{i}"
        df[f"rank_{i}"] = df[col].rank(method="min", ascending=False) - 1

    n = len(df)
    df["mean"] = df[[f"rank_{i}" for i in range(num_importance)]].mean(axis=1) / n
    df["std"] = df[[f"rank_{i}" for i in range(num_importance)]].std(axis=1) / n
    return df


dfs = []
for layer in range(binn.n_layers):
    layer_df = importance_df_copy[
        importance_df_copy["source_layer"] == layer
    ].copy()
    layer_df = compute_normalized_ranks(layer_df, num_importance=3)
    dfs.append(layer_df[["source_node", "source_layer", "mean", "std"]])

plot_df = (
    pd.concat(dfs, ignore_index=True)
    .rename(columns={"source_node": "source", "source_layer": "source layer"})
)

# top30
# Out[200]:
#      source  source layer      mean       std
# 411  Q13418             0  0.002341  0.002924
# 353  P60709             0  0.007022  0.003716
# 535  Q8NGJ1             0  0.017790  0.016932
# 187  P11802             0  0.018258  0.011496
# 313  P49005             0  0.020131  0.011695
# 219  P18858             0  0.020131  0.020562
# 650  Q9NQT4             0  0.022472  0.035280
# 517  Q86WU2             0  0.022940  0.018330
# 6    O00168             0  0.023408  0.023516
# 664  Q9NVD7             0  0.023408  0.014753
# 15   O14503             0  0.023876  0.035280
# 199  P14136             0  0.026217  0.018861
# 26   O15111             0  0.027622  0.024582
# 699  Q9Y4C1             0  0.029494  0.017029
# 527  Q8N257             0  0.030431  0.018276
# 310  P48059             0  0.035112  0.019663
# 245  P24941             0  0.040262  0.037062
# 467  Q16576             0  0.041199  0.057983
# 549  Q8WWM9             0  0.044944  0.028508
# 149  P07951             0  0.048221  0.038523
# 246  P25067             0  0.048689  0.019879
# 108  P00918             0  0.051966  0.024367
# 161  P08670             0  0.052434  0.023931
# 253  P27658             0  0.054775  0.013398
# 55   O43914             0  0.058052  0.036634
# 388  Q05682             0  0.059925  0.007069
# 276  P35052             0  0.060393  0.052004
# 141  P06493             0  0.062266  0.017163
# 43   O43521             0  0.065543  0.021773
# 509  Q7Z4I7             0  0.065543  0.037485
# top30_final
# Out[201]:
# ['ILK',
#  'ACTB',
#  'OR4D6',
#  'CDK4',
#  'POLD2',
#  'LIG1',
#  'EXOSC5',
#  'LDHD',
#  'FXYD1',
#  'PARVA',
#  'BHLHE40',
#  'GFAP',
#  'CHUK',
#  'KDM3A',
#  'H2BC26',
#  'LIMS1',
#  'CDK2',
#  'RBBP7',
#  'CYGB',
#  'TPM2',
#  'COL8A2',
#  'CA2',
#  'VIM',
#  'COL8A1',
#  'TYROBP',
#  'CALD1',
#  'GPC1',
#  'CDK1',
#  'BCL2L11',
#  'LIMS2']