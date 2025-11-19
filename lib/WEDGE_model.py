import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
import pytorch_lightning as pl
import torchmetrics
import math
from torch_geometric.nn import global_mean_pool, global_max_pool, GraphNorm
class HeteroGCN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate=0.5, dp_rate_linear=0.5, **kwargs):
        super().__init__()

        self.conv1 = HeteroConv({
            ('protein', 'interacts', 'protein'): GCNConv(c_in, c_hidden, add_self_loops=True),
            ('gene', 'regulates', 'gene'): GCNConv(c_in, c_hidden, add_self_loops=True)
            # ('protein', 'associates', 'gene'): GCNConv(c_in, c_hidden),
            # ('gene', 'associates', 'protein'): GCNConv(c_in, c_hidden),
        })

        self.conv2 = HeteroConv({
            ('protein', 'interacts', 'protein'): GCNConv(c_hidden, c_hidden, add_self_loops=True),
            ('gene', 'regulates', 'gene'): GCNConv(c_hidden, c_hidden, add_self_loops=True)
        })

        self.norm_dict1 = nn.ModuleDict({
            'protein': GraphNorm(c_hidden),
            'gene': GraphNorm(c_hidden)
        })

        self.norm_dict2 = nn.ModuleDict({
            'protein': GraphNorm(c_hidden),
            'gene': GraphNorm(c_hidden)
        })

        self.dropout = nn.Dropout(dp_rate)

        self.protein_head = nn.Sequential(
            nn.Linear(c_hidden * 2, c_hidden),
            nn.LayerNorm(c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

        self.gene_head = nn.Sequential(
            nn.Linear(c_hidden * 2, c_hidden),
            nn.LayerNorm(c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

        self.attention = nn.Sequential(
            nn.Linear(c_out * 2, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x_dict, edge_index_dict, batch=None):
        if batch is None:
            batch = {
                'protein': torch.zeros(x_dict['protein'].size(0), dtype=torch.long, device=x_dict['protein'].device),
                'gene': torch.zeros(x_dict['gene'].size(0), dtype=torch.long, device=x_dict['gene'].device)
            }

        x_dict1 = self.conv1(x_dict, edge_index_dict)

        x_dict1 = {
            node_type: self.norm_dict1[node_type](x, batch[node_type])
            for node_type, x in x_dict1.items()
        }

        x_dict1 = {key: F.relu(x) for key, x in x_dict1.items()}
        x_dict1 = {key: self.dropout(x) for key, x in x_dict1.items()}

        x_dict2 = self.conv2(x_dict1, edge_index_dict)
        x_dict2 = {
            node_type: self.norm_dict2[node_type](x, batch[node_type])
            for node_type, x in x_dict2.items()
        }

        x_dict2 = {
            key: x + x_dict1[key]
            for key, x in x_dict2.items()
        }

        x_dict2 = {key: F.relu(x) for key, x in x_dict2.items()}
        x_dict2 = {key: self.dropout(x) for key, x in x_dict2.items()}

        x_protein = x_dict2['protein']
        protein_mean = global_mean_pool(x_protein, batch['protein'])
        protein_max = global_max_pool(x_protein, batch['protein'])
        protein_features = torch.cat([protein_mean, protein_max], dim=1)
        protein_out = self.protein_head(protein_features)

        x_gene = x_dict2['gene']
        gene_mean = global_mean_pool(x_gene, batch['gene'])
        gene_max = global_max_pool(x_gene, batch['gene'])
        gene_features = torch.cat([gene_mean, gene_max], dim=1)
        gene_out = self.gene_head(gene_features)

        combined = torch.cat([protein_out, gene_out], dim=1)
        attention_weights = self.attention(combined)

        return {
            'protein_out': protein_out,
            'gene_out': gene_out,
            'attention_weights': attention_weights,
            'combined_out': attention_weights[:, 0:1] * protein_out +
                            attention_weights[:, 1:2] * gene_out
        }
def get_scheduler_with_warmup(optimizer, num_warmup_steps, scheduler_type="linear"):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
class GraphLevelHeteroGCN(pl.LightningModule):
    def __init__(self, c_in, c_hidden, c_out, lr=1e-3, weight_decay=1e-3,
                 dp_rate=0.5, dp_rate_linear=0.5,
                 warmup_steps=1000,
                 label_smoothing=0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['train_loader'])

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.model = HeteroGCN(
            c_in=c_in, c_hidden=c_hidden, c_out=c_out,
            dp_rate=dp_rate, dp_rate_linear=dp_rate_linear,
            **kwargs
        )

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        metrics = {}
        for prefix in ['protein', 'gene', 'combined']:
            metrics[f'{prefix}_accuracy'] = torchmetrics.Accuracy(num_classes=c_out)
            metrics[f'{prefix}_f1'] = torchmetrics.F1Score(num_classes=c_out, average='macro')
        self.metrics = nn.ModuleDict(metrics)

    def _compute_metrics(self, outputs, batch_y, stage):
        batch_y = batch_y.long()
        losses = {}


        # protein_attention = outputs['attention_weights'][:, 0].mean()
        # gene_attention = outputs['attention_weights'][:, 1].mean()
        # self.log(f'{stage}_protein_attention', protein_attention, prog_bar=True, batch_size=len(batch_y))
        # self.log(f'{stage}_gene_attention', gene_attention, prog_bar=True, batch_size=len(batch_y))

        losses['protein_loss'] = self.loss(outputs['protein_out'], batch_y)
        losses['gene_loss'] = self.loss(outputs['gene_out'], batch_y)
        losses['combined_loss'] = self.loss(outputs['combined_out'], batch_y)

        total_loss = (losses['protein_loss'] * 0.4 +
                      losses['gene_loss'] * 0.4 +
                      losses['combined_loss'] * 0.2)
        self.log(f'{stage}_total_loss', total_loss, prog_bar=True, batch_size=len(batch_y))
        # 其他指标计算和记录...
        with torch.no_grad():
            predictions = {
                'protein': torch.argmax(F.softmax(outputs['protein_out'], dim=1), dim=1),
                'gene': torch.argmax(F.softmax(outputs['gene_out'], dim=1), dim=1),
                'combined': torch.argmax(F.softmax(outputs['combined_out'], dim=1), dim=1)
            }

        for prefix in ['protein', 'gene', 'combined']:
            acc = self.metrics[f'{prefix}_accuracy'](predictions[prefix], batch_y)
            f1 = self.metrics[f'{prefix}_f1'](predictions[prefix], batch_y)

            self.log(f'{stage}_{prefix}_loss', losses[f'{prefix}_loss'],
                     prog_bar=True, batch_size=len(batch_y))
            self.log(f'{stage}_{prefix}_acc', acc,
                     prog_bar=True, batch_size=len(batch_y))
            self.log(f'{stage}_{prefix}_f1', f1,
                     prog_bar=True, batch_size=len(batch_y))

        return total_loss

    def forward(self, x_dict, edge_index_dict, batch_dict):
        return self.model(x_dict, edge_index_dict, batch_dict)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        total_loss = self._compute_metrics(outputs, batch.y, 'train')
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        self._compute_metrics(outputs, batch.y, 'val')

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        return self._compute_metrics(outputs, batch.y, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        lr_scheduler = {
            "scheduler": get_scheduler_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                scheduler_type="linear"
            ),
            "monitor": "val_total_loss",
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
