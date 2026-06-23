import torch
import pandas as pd
from torch_geometric.data import DataLoader


class NodeImportanceAnalyzer:
    """
    对 WEDGE 异构图模型计算节点重要性。
    仅支持 Integrated Gradients 方法。
    """

    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def analyze_importance(self, data, target_class):
        return {
            'integrated_gradients': self._compute_integrated_gradients(data, target_class),
        }

    # -------------------------------------------------------------------------
    # Integrated Gradients：逐点积分
    # -------------------------------------------------------------------------
    def _compute_integrated_gradients(self, data, target_class):
        """手动实现的 Integrated Gradients。"""
        baseline_dict = {
            key: torch.zeros_like(data.x_dict[key])
            for key in data.x_dict
        }

        alphas = torch.linspace(0, 1, self.steps)
        total_gradients = {
            key: torch.zeros_like(data.x_dict[key])
            for key in data.x_dict
        }

        for alpha in alphas:
            current_input = {}
            for key in data.x_dict:
                current_input[key] = (
                    baseline_dict[key]
                    + alpha * (data.x_dict[key] - baseline_dict[key])
                )
                current_input[key].requires_grad_(True)

            self.model.eval()
            outputs = self.model(current_input, data.edge_index_dict, data.batch_dict)
            score = outputs['combined_out'][:, target_class].sum()

            gradients = torch.autograd.grad(
                score,
                [current_input[k] for k in ['protein', 'gene']],
            )
            for k, g in zip(['protein', 'gene'], gradients):
                total_gradients[k] += g

        attributions = {}
        for node_type in ['protein', 'gene']:
            attributions[node_type] = (
                (data.x_dict[node_type] - baseline_dict[node_type])
                * total_gradients[node_type]
                / self.steps
            ).abs().mean(dim=1)

        return attributions


def analyze_node_importance(model, data, target_class):
    analyzer = NodeImportanceAnalyzer(model)
    importance_scores = analyzer.analyze_importance(data, target_class)
    results = {}
    for method in importance_scores:
        results[method] = {}
        for node_type in ['protein', 'gene']:
            scores = importance_scores[method][node_type]
            results[method][node_type] = {
                f"{node_type}_{i}": score.item()
                for i, score in enumerate(scores)
            }
    return results


def add_genename(results, gene_names):
    for method in ['integrated_gradients']:
        for node_type in ['protein', 'gene']:
            results[method][node_type] = {
                gene_names[i]: score
                for i, score in enumerate(results[method][node_type].values())
            }
    return results


def HGCN_Node_Importance_Explainer(
    model, gene_names, test_dataset, explain_type='integrated_gradients'
):
    """
    对异构图 WEDGE 模型进行可解释性分析。

    Parameters
    ----------
    model : GraphLevelHeteroGCN
    gene_names : list of str
    test_dataset : HeteroDataset
    explain_type : str
        'integrated_gradients' — 手动实现的 IG（推荐，稳定）

    Returns
    -------
    df_ave_label0_protein, df_ave_label0_gene,
    df_ave_label1_protein, df_ave_label1_gene : pd.Series
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_loader_t = DataLoader(test_dataset, batch_size=1, shuffle=False)
    label_0_importance = []
    label_1_importance = []

    for batch_data in test_loader_t:
        batch_data = batch_data.to(device)

        if batch_data.y == 0:
            target_class = int(batch_data.y.item())
            label_0 = analyze_node_importance(model, batch_data, target_class)
            label_0 = add_genename(label_0, gene_names)
            label_0_importance.append(label_0)

        if batch_data.y == 1:
            target_class = int(batch_data.y.item())
            label_1 = analyze_node_importance(model, batch_data, target_class)
            label_1 = add_genename(label_1, gene_names)
            label_1_importance.append(label_1)

    if explain_type != 'integrated_gradients':
        raise ValueError(
            f"Unsupported explain_type: {explain_type}. "
            f"Only 'integrated_gradients' is supported."
        )

    # 聚合
    def aggregate(entry_list, key):
        protein_data = [entry[key]['protein'] for entry in entry_list]
        gene_data = [entry[key]['gene'] for entry in entry_list]

        df_p = pd.DataFrame(protein_data)
        df_p.index = [f"Patient_{i + 1}" for i in range(len(protein_data))]
        df_p_tensor = torch.tensor(df_p.values, dtype=torch.float32).to(device)
        ave_protein = pd.Series(
            df_p_tensor.mean(dim=0).cpu().numpy(), index=df_p.columns
        )

        df_g = pd.DataFrame(gene_data)
        df_g.index = [f"Patient_{i + 1}" for i in range(len(gene_data))]
        df_g_tensor = torch.tensor(df_g.values, dtype=torch.float32).to(device)
        ave_gene = pd.Series(
            df_g_tensor.mean(dim=0).cpu().numpy(), index=df_g.columns
        )

        return ave_protein, ave_gene

    df_ave_label0_protein, df_ave_label0_gene = aggregate(label_0_importance, 'integrated_gradients')
    df_ave_label1_protein, df_ave_label1_gene = aggregate(label_1_importance, 'integrated_gradients')

    return (
        df_ave_label0_protein,
        df_ave_label0_gene,
        df_ave_label1_protein,
        df_ave_label1_gene,
    )
