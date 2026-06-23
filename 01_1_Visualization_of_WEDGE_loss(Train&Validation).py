# ==============================================================================
# 01_1: 可视化 WEDGE 训练过程的 Loss 曲线（EMA 平滑+多 Fold 叠加优化版）
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 配置
# ==============================================================================
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR
LIGHTNING_LOG_DIR = os.path.join(PROJECT_ROOT, 'lightning_logs')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'fig', 'loss_curves')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 个 Fold 的学术专属配色
FOLD_COLORS = ['#E64B34', '#3C5487', '#00A088', '#F4AE64', '#4DBBD6']

# 全局绘图样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42  # 保持矢量字体可编辑性
})

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42  # 保持矢量字体可编辑性
})


# ==============================================================================
# 辅助函数：学术轴设置与 EMA 平滑
# ==============================================================================
def apply_academic_axis(ax):
    ax.tick_params(axis='both', which='major', direction='out',
                   top=False, right=False, width=1.5, length=6, labelsize=11)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def data_smoothing(values, weight=0.85):
    """指数移动平均 (EMA) 平滑算法"""
    if len(values) == 0:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ==============================================================================
# 读取数据
# ==============================================================================
def load_loss_data(fold: int):
    from tensorboard.backend.event_processing import event_accumulator
    log_path = os.path.join(LIGHTNING_LOG_DIR, f'heterogcn_{fold}', 'version_0')
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    def get_series(tag):
        try:
            events = ea.Scalars(tag)
            return [e.step for e in events], [e.value for e in events]
        except Exception:
            return [], []

    data = {}
    for prefix in ['train', 'val']:
        for loss_type in ['total', 'protein', 'gene', 'combined']:
            tag = f'{prefix}_{loss_type}_loss'
            steps, values = get_series(tag)
            data[f'{prefix}_{loss_type}_loss'] = (steps, values)
    return data


# ==============================================================================
# 核心核心优化绘图函数
# ==============================================================================
def plot_single_metric_overlay(all_data, data_key, title, filename, save_dir, ylim=None, smooth_w=0.85):
    """
    绘制单张拉长图。通过 EMA 算法分离‘高频噪声背景’与‘核心趋势曲线’。
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))

    for fold_idx, (fold, data) in enumerate(sorted(all_data.items())):
        color = FOLD_COLORS[fold_idx]
        steps, values = data[data_key]

        if steps:
            # 1. 绘制极淡的原始毛刺数据作为背景（防信息丢失，增加图表细节层次）
            ax.plot(steps, values, color=color, linewidth=0.6, alpha=0.15)

            # 2. 绘制高饱和度的 EMA 平滑趋势线（作为视觉核心）
            smoothed_values = data_smoothing(values, weight=smooth_w)
            ax.plot(steps, smoothed_values, color=color, linewidth=1.8,
                    alpha=0.95, label=f'Fold {fold}')

    # 规范化学术坐标轴
    apply_academic_axis(ax)
    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='normal', pad=12)

    # 经典细黑边图例
    ax.legend(fontsize=10, frameon=True, edgecolor='black', framealpha=0.9, loc='upper right')

    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()
    plt.close(fig)


def save_loss_data_to_csv(all_data, save_dir):
    """
    将 all_data 保存为 CSV 文件，每个 Fold 一个文件：
    fold_{n}_loss.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    loss_keys = [
        'train_total_loss', 'train_protein_loss', 'train_gene_loss', 'train_combined_loss',
        'val_total_loss', 'val_protein_loss', 'val_gene_loss', 'val_combined_loss'
    ]
    for fold, data in sorted(all_data.items()):
        # 取所有 key 的公共最小长度，避免不等长问题
        lengths = [len(data[k][1]) for k in loss_keys if data[k][1]]
        min_len = min(lengths) if lengths else 0
        rows = {'step': data['train_total_loss'][0][:min_len]}
        for key in loss_keys:
            rows[key] = data[key][1][:min_len]
        df = pd.DataFrame(rows)
        out_path = os.path.join(save_dir, f'fold_{fold}_loss.csv')
        df.to_csv(out_path, index=False)
        print(f'Saved: {out_path}  ({len(df)} rows)')


# ==============================================================================
# 主逻辑
# ==============================================================================
if __name__ == '__main__':
    all_data = {}
    for fold in range(1, 6):
        try:
            data = load_loss_data(fold)
            all_data[fold] = data
            print(f'  Fold {fold} loaded.')
        except Exception as e:
            print(f'  Fold {fold} failed: {e}')

    print('\nGenerating optimized smoothed plots...\n')

    # Save loss data to CSV
    save_loss_data_to_csv(all_data, OUTPUT_DIR)

    # 1. Train Total Loss (适当平滑)
    plot_single_metric_overlay(
        all_data=all_data, data_key='train_total_loss',
        title='Training Total Loss — All Folds Overlay',
        filename='train_total_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.1, 1), smooth_w=0.9
    )

    # 2. Train PPI Loss
    plot_single_metric_overlay(
        all_data=all_data, data_key='train_protein_loss',
        title='Training PPI Loss — All Folds Overlay',
        filename='train_PPI_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.1, 1), smooth_w=0.9
    )

    # 3. Train GRN Loss (纵坐标限制为 0.1 ~ 1，调高平滑度至 0.90 压制强噪声)
    plot_single_metric_overlay(
        all_data=all_data, data_key='train_gene_loss',
        title='Training GRN Loss — All Folds Overlay',
        filename='train_GRN_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.1, 1), smooth_w=0.90
    )

    # 4. Val Total Loss (验证集波动一般较小，平滑度设为 0.7 即可)
    plot_single_metric_overlay(
        all_data=all_data, data_key='val_total_loss',
        title='Validation Total Loss — All Folds Overlay',
        filename='val_total_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.2, 1),smooth_w=0.9
    )

    # 5. Val PPI Loss
    plot_single_metric_overlay(
        all_data=all_data, data_key='val_protein_loss',
        title='Validation PPI Loss — All Folds Overlay',
        filename='val_PPI_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.2, 1),smooth_w=0.9
    )

    # 6. Val GRN Loss (纵坐标限制为 0.3 ~ 1)
    plot_single_metric_overlay(
        all_data=all_data, data_key='val_gene_loss',
        title='Validation GRN Loss — All Folds Overlay',
        filename='val_GRN_loss.pdf', save_dir=OUTPUT_DIR,
        ylim=(0.2, 1), smooth_w=0.70
    )

    print('\nAll 6 smoothed loss curves saved to:', OUTPUT_DIR)