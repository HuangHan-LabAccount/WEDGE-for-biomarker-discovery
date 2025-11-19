from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data
import os
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def SplitData(matrix, meta):
    meta_train = meta[meta['Batch'] == 1]
    meta_test = meta[meta['Batch'] == 2]
    matrix_train = matrix.loc[meta_train['MS_number']]
    matrix_test = matrix.loc[meta_test['MS_number']]
    return matrix_train, matrix_test, meta_train, meta_test
def load_Stringdatabase(path = "H:/Proteomic/PPI_data/String_database", file_name = "human_PPI_score_Stringdatabase(700up).csv"):
    protein_matrix = pd.read_csv(os.path.join(path, file_name))
    return protein_matrix
def load_dataset(path = "H:/Proteomic"):
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    # patient * gene
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    # HPV_EA_DEG = pd.read_csv(f'{path}/data/DEG/HPV_EA_DEG.csv', index_col=0)
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)

    # adj matrix
    protein_score = load_Stringdatabase(path=os.path.join(path, 'PPI_data/String_database/'))
    adj_norm = getAdjByString(protein_score, matrix_train)

    # Standardize
    # patient * gene
    std = StandardScaler()
    X_train = matrix_train.to_numpy()
    X_test = matrix_test.to_numpy()
    X_train = std.fit_transform(X_train).T
    X_test = std.transform(X_test).T
    encoder = LabelEncoder()

    # CancerType
    y_train = meta_train.CancerType
    y_test = meta_test.CancerType
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    gene_list = matrix.columns
    return X_train, X_test, y_train, y_test, gene_list, matrix, adj_norm
def getAdjByString(protein_matrix, data_matrix, one_direction = False):
    if not one_direction:
        edges_unordered = protein_matrix[['protein1', 'protein2', 'combined_score']].values
        idx = []
        # gene * patient
        matrix = data_matrix.T
        for i in range(len(edges_unordered)):
            if edges_unordered[i, 0] in matrix.index and edges_unordered[i, 1] in matrix.index:
                idx.append(i)
        edges_unordered = edges_unordered[idx]
        idx = np.array(matrix.index)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(
            edges_unordered[:, :2].shape)
        weights = edges_unordered[:, 2].astype(np.float32)
        adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                            shape=(matrix.shape[0], matrix.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj)
    else:
        edges_unordered = protein_matrix[["TF", "Target"]].values  # 保留单向信息
        idx = []
        matrix = data_matrix.T
        for i in range(len(edges_unordered)):
            if edges_unordered[i, 0] in matrix.index and edges_unordered[i, 1] in matrix.index:
                idx.append(i)
        edges_unordered = edges_unordered[idx]
        idx = np.array(matrix.index)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(
            edges_unordered[:, :2].shape)
        weights = np.ones(edges_unordered.shape[0], dtype=np.float32)
        adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                            shape=(matrix.shape[0], matrix.shape[0]),
                            dtype=np.float32)
    return adj
def build_hetero_graph_dataset(matrix,adj_PPI,adj_GRN,labels):
    dataset = []
    # Convert DataFrame to numpy array if it's a DataFrame
    if isinstance(matrix, pd.DataFrame):
        matrix_array = matrix.to_numpy()
    else:
        matrix_array = matrix

    num_rows = matrix_array.shape[0]

    # Convert PPI adjacency matrix to torch tensor if it's not already
    if not isinstance(adj_PPI, torch.Tensor):
        adj_PPI_tensor = torch.tensor(adj_PPI.toarray(), dtype=torch.float)
    else:
        adj_PPI_tensor = adj_PPI

    if not isinstance(adj_GRN, torch.Tensor):
        adj_GRN_tensor = torch.tensor(adj_GRN.toarray(), dtype=torch.float)
    else:
        adj_GRN_tensor = adj_GRN

    edge_index_GRN, edge_weight_GRN = dense_to_sparse(adj_GRN_tensor)
    edge_index_PPI, edge_weight_PPI = dense_to_sparse(adj_PPI_tensor)
    for i in range(num_rows):
        # Create heterogeneous graph data object
        data = HeteroData()

        # Add node features (only store once and reference for both types)
        node_features = matrix_array[i].reshape(-1, 1)  # (num_nodes, 1)
        x = torch.tensor(node_features, dtype=torch.float)
        data['protein'].x = x
        data['gene'].x = x  # This will reference the same tensor

        # Add PPI edges with weights
        data['protein', 'interacts', 'protein'].edge_index = edge_index_PPI
        data['protein', 'interacts', 'protein'].edge_weight = edge_weight_PPI

        # Add GRN edges (all weights are implicitly 1)
        data['gene', 'regulates', 'gene'].edge_index = edge_index_GRN
        data['gene', 'regulates', 'gene'].edge_weight = edge_weight_GRN
        # Add label as a one-dimensional tensor
        if isinstance(labels, (pd.Series, pd.DataFrame)):
            label = labels.iloc[i]
        else:
            label = labels[i]
        data.y = torch.tensor([label], dtype=torch.float)

        dataset.append(data)

    return dataset
def DEgene_selected(matrix, path = "H:/Proteomic"):
    DE_gene = pd.read_csv(f'{path}/data/DEG/HPV_EA_DEG_1.5.csv', index_col=0)
    return matrix.loc[:, DE_gene.x]