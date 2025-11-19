import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import DataLoader

class NodeImportanceAnalyzer:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps
        self.activations = {}
        self.gradients = {}
        def forward_hook(name):
            def hook(module, input, output):
                if 'conv2' not in self.activations:
                    self.activations['conv2'] = {}
                self.activations['conv2'][name] = output
            return hook
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if 'conv2' not in self.gradients:
                    self.gradients['conv2'] = {}
                self.gradients['conv2'][name] = grad_output[0]
            return hook
        self.model.model.conv2.convs[('protein', 'interacts', 'protein')].register_forward_hook(
            forward_hook('protein')
        )
        self.model.model.conv2.convs[('gene', 'regulates', 'gene')].register_forward_hook(
            forward_hook('gene')
        )
        self.model.model.conv2.convs[('protein', 'interacts', 'protein')].register_backward_hook(
            backward_hook('protein')
        )
        self.model.model.conv2.convs[('gene', 'regulates', 'gene')].register_backward_hook(
            backward_hook('gene')
        )
    def analyze_importance(self, data, target_class):
        return {
            'integrated_gradients': self._compute_integrated_gradients(data, target_class),
            'gradcam': self._compute_gradcam(data, target_class)
        }
    def _compute_integrated_gradients(self, data, target_class):
        baseline_dict = {}
        for key in data.x_dict:
            baseline_dict[key] = torch.zeros_like(data.x_dict[key])
        alphas = torch.linspace(0, 1, self.steps)
        total_gradients = {key: torch.zeros_like(data.x_dict[key]) for key in data.x_dict}
        for alpha in alphas:
            current_input = {}
            for key in data.x_dict:
                current_input[key] = baseline_dict[key] + alpha * (data.x_dict[key] - baseline_dict[key])
                current_input[key].requires_grad_(True)
            self.model.eval()
            outputs = self.model(current_input, data.edge_index_dict, data.batch_dict)
            score = outputs['combined_out'][:, target_class].sum()
            gradients = torch.autograd.grad(score, [current_input[k] for k in ['protein', 'gene']])
            for k, g in zip(['protein', 'gene'], gradients):
                total_gradients[k] += g
        attributions = {}
        for node_type in ['protein', 'gene']:
            attributions[node_type] = (
                    (data.x_dict[node_type] - baseline_dict[node_type]) *
                    total_gradients[node_type] / self.steps
            ).abs().mean(dim=1)
        return attributions
    def _compute_gradcam(self, data, target_class):
        self.model.eval()
        outputs = self.model(data.x_dict, data.edge_index_dict, data.batch_dict)
        score = outputs['combined_out'][:, target_class].sum()
        self.model.zero_grad()
        score.backward()
        importance_scores = {}
        for node_type in ['protein', 'gene']:
            activations = self.activations['conv2'][node_type]
            gradients = self.gradients['conv2'][node_type]
            weights = torch.mean(gradients, dim=0)
            cam = torch.sum(weights.unsqueeze(0) * activations, dim=1)
            cam = F.relu(cam)  # 只保留正面贡献
            if cam.max() > 0:
                cam = cam / cam.max()
            importance_scores[node_type] = cam
        return importance_scores
def analyze_node_importance(model, data, target_class):
    analyzer = NodeImportanceAnalyzer(model)
    importance_scores = analyzer.analyze_importance(data, target_class)
    results = {}
    for method in importance_scores:
        results[method] = {}
        for node_type in ['protein', 'gene']:
            scores = importance_scores[method][node_type]
            node_importance = {
                f"{node_type}_{i}": score.item()
                for i, score in enumerate(scores)
                }
            results[method][node_type] = node_importance
    return results
def add_genename(results, gene_names):
    for method in ['integrated_gradients', 'gradcam']:
        for node_type in ['protein', 'gene']:
            results[method][node_type] = {
                gene_names[i]: score
                for i, score in enumerate(results[method][node_type].values())
            }
    return results
def HGCN_Node_Importance_Explianer(model, gene_names, test_dataset, explain_type='integrated_gradients'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    test_loader_t = DataLoader(test_dataset, batch_size=1, shuffle=False)
    label_0_importance = []
    label_1_importance = []

    for batch_idx, batch_data in enumerate(test_loader_t):
        batch_data = batch_data.to(device)

        if batch_data.y == 0:
            target_class = torch.tensor(batch_data.y.cpu().numpy(), dtype=torch.int).to(device)
            label_0 = analyze_node_importance(model, batch_data, target_class=target_class)
            label_0 = add_genename(label_0, gene_names)
            label_0_importance.append(label_0)

        if batch_data.y == 1:
            target_class = torch.tensor(batch_data.y.cpu().numpy(), dtype=torch.int).to(device)
            label_1 = analyze_node_importance(model, batch_data, target_class=target_class)
            label_1 = add_genename(label_1, gene_names)
            label_1_importance.append(label_1)

    if explain_type == 'integrated_gradients':
        # label_0 integrated_gradients
        protein_data = [entry['integrated_gradients']['protein'] for entry in label_0_importance]
        df = pd.DataFrame(protein_data)
        df.index = [f"Patient_{i + 1}" for i in range(len(protein_data))]
        df_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        df_ave_label0_protein = pd.Series(df_tensor.mean(dim=0).cpu().numpy(), index=df.columns)

        gene_data = [entry['integrated_gradients']['gene'] for entry in label_0_importance]
        df = pd.DataFrame(gene_data)
        df.index = [f"Patient_{i + 1}" for i in range(len(gene_data))]
        df_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        df_ave_label0_gene = pd.Series(df_tensor.mean(dim=0).cpu().numpy(), index=df.columns)

        protein_data = [entry['integrated_gradients']['protein'] for entry in label_1_importance]
        df = pd.DataFrame(protein_data)
        df.index = [f"Patient_{i + 1}" for i in range(len(protein_data))]
        df_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        df_ave_label1_protein = pd.Series(df_tensor.mean(dim=0).cpu().numpy(), index=df.columns)

        gene_data = [entry['integrated_gradients']['gene'] for entry in label_1_importance]
        df = pd.DataFrame(gene_data)
        df.index = [f"Patient_{i + 1}" for i in range(len(gene_data))]
        df_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        df_ave_label1_gene = pd.Series(df_tensor.mean(dim=0).cpu().numpy(), index=df.columns)

    return df_ave_label0_protein, df_ave_label0_gene, df_ave_label1_protein, df_ave_label1_gene

