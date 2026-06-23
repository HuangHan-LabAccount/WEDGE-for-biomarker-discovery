# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP
# https://pmc.ncbi.nlm.nih.gov/articles/PMC8910329/#notes4
# the WGANGP code is based on the above link
# the generate_data function is used to generate new data for the label 0 and label 1
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = 'E:/WEDGE_article/Proteomic/WEDGE_code'
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(os.path.join(PROJECT_ROOT, 'lib/'))
from utilsdata import *

# Data paths
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'Aug_data')
PPI_DIR = os.path.join(PROJECT_ROOT, 'PPI_GRN_database')
STRING_DIR = os.path.join(PPI_DIR, 'String_database')
TRRUST_DIR = os.path.join(PPI_DIR, 'Trrust_database')

# Label encoding: 0 = HPV_related, 1 = NHPV
LABEL_HPV = 0
LABEL_NHPV = 1


class GeneExpressionDataset(Dataset):
    def __init__(self, data, genes, train=False):
        self.data = data
        self.genes = genes
        self.X = self.data[self.genes].values.astype('float32')
        self.y = self.data.label.values.astype('float32')

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        return sample

    def __len__(self):
        return len(self.data)


def gradient_penalty(critic, real, fake):
    BATCHSIZE, GENES = real.shape
    epsilon = torch.rand((BATCHSIZE, 1)).repeat(1, GENES).cuda()
    interpolated_arrays = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(interpolated_arrays)
    gradient = torch.autograd.grad(
        inputs=interpolated_arrays,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)


def generate_data(matrix, label_list, label, savepath, niter=10000, **kwargs):
    """
    Generate augmented data using WGAN-GP for a specific label.

    Parameters:
    -----------
    matrix : pd.DataFrame
        Gene expression matrix (samples x genes)
    label_list : array-like
        Labels for each sample
    label : int
        The label (0 or 1) to generate augmented data for
    savepath : str
        Directory to save the generated data
    niter : int
        Number of epochs to train
    kwargs : dict
        Additional parameters:
        - f : int, number of folds for cross-validation (default 5)
        - augFactor : int, augmentation factor (default 5)
        - z_dim : int, size of latent vector z (default 100)
        - batchSize : int, batch size (default 16)
        - critic_iterations : int, number of critic iterations (default 3)
        - learning_rate : float, learning rate (default 5e-5)
        - lambda_gp : float, gradient penalty weight (default 10)
    """
    f = kwargs.get('f', 5)
    augFactor = kwargs.get('augFactor', 5)
    z_dim = kwargs.get('z_dim', 100)
    batchSize = kwargs.get('batchSize', 16)
    niter = kwargs.get('niter', niter)
    critic_iterations = kwargs.get('critic_iterations', 3)
    learning_rate = kwargs.get('learning_rate', 5e-5)
    lambda_gp = kwargs.get('lambda_gp', 10)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=f, shuffle=True, random_state=42)

    label_list = np.array(label_list)
    for fold, (train_index, val_index) in enumerate(skf.split(matrix, label_list), 1):
        print(f"Processing fold {fold}...")
        df = matrix.copy()
        genes = df.select_dtypes('float').columns
        df.insert(0, 'label', label_list)
        df.insert(0, 'id', df.index)
        df.insert(0, 'subset', 'test')
        df.loc[df.index[train_index], 'subset'] = 'train'
        df.loc[df.index[val_index], 'subset'] = 'test'
        DF = df

        gandata = DF[(DF.subset == 'train') & (DF.label == label)]
        X_dim = len(genes)

        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.main = nn.Sequential(
                    nn.Linear(z_dim, 250, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(250, 500, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(500, 1000, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(1000, X_dim, bias=True),
                    nn.Tanh()
                )

            def forward(self, input):
                output = self.main(input)
                return output

        class Critic(nn.Module):
            def __init__(self):
                super(Critic, self).__init__()
                self.main = nn.Sequential(
                    nn.Linear(X_dim, 1000),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(1000, 500),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(500, 250),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(250, 1),
                )

            def forward(self, input):
                output = self.main(input)
                return output.squeeze(1)

        print(f'Generate new data for label {label}; train for {niter} epochs')

        # Initialize Generator and Critic
        generator = Generator().to(device)
        generator.apply(weights_init)

        critic = Critic().to(device)
        critic.apply(weights_init)

        optimizerC = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

        # Scale data
        scaler.fit(gandata[genes])
        gandata_sc = scaler.transform(gandata[genes])
        gandata_sc = pd.DataFrame(gandata_sc, index=gandata.index, columns=genes)
        gandata_sc.insert(0, 'label', gandata.label)
        gandata_sc.insert(0, 'id', gandata.id)
        gandata_sc.insert(0, 'subset', gandata.subset)

        dataset = GeneExpressionDataset(gandata_sc, genes, True)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)

        nnew = len(gandata) * augFactor
        fixed_noise = torch.randn(nnew, z_dim).cuda()
        generator.train().to(device)
        critic.train().to(device)

        # Training Loop
        for epoch in range(niter):
            for i, data in enumerate(dataloader, 0):
                real = data[0].to(device)
                for _ in range(critic_iterations):
                    noise = torch.randn(real.size(0), z_dim).to(device)
                    fake = generator(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake).reshape(-1)
                    gp = gradient_penalty(critic, real, fake)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                    critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    optimizerC.step()
                noise = torch.randn(batchSize, z_dim).to(device)
                fake = generator(noise)
                output = critic(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                generator.zero_grad()
                loss_gen.backward()
                optimizerG.step()
            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch + 1 == niter:
                print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch + 1, niter, loss_critic.item(), loss_gen.item()))

        # Generate and save new data
        with torch.no_grad():
            fixed_noise = fixed_noise.to(device)
            fakes = generator(fixed_noise).cpu().detach()
        gen_np = fakes.numpy()
        gen_np = scaler.inverse_transform(gen_np)
        gen_df = pd.DataFrame(gen_np)
        gen_df.columns = genes
        idxnames = ['fold{}_a{}_label{}'.format(fold, i + 1, label) for i in range(nnew)]
        gen_df.insert(0, 'label', [label] * nnew)
        gen_df.insert(0, 'id', idxnames)
        gen_df.insert(0, 'subset', ['train'] * nnew)
        save_file = f'{savepath}/generated_data_fold{fold}_{label}.csv'
        gen_df.to_csv(save_file, index=False)
        print(f'Fold {fold} data saved to {save_file}')


if __name__ == '__main__':
    # WGAN-GP parameters (consistent with original WGANGP.py)
    kwargs = {
        'method': 'WGAN-GP',
        'f': 5,                      # fold of cross validation
        'augFactor': 5,              # augmentation factor
        'z_dim': 100,                # size of the latent vector z
        'batchSize': 16,             # batch size
        'niter': 10000,             # number of epochs to train for
        'critic_iterations': 3,     # number of critic iterations before generator trains
        'learning_rate': 5e-5,       # learning rate
        'lambda_gp': 10             # 'weight' of the gradient penalty
    }

    # Create output directory if it doesn't exist
    os.makedirs(AUG_DATA_DIR, exist_ok=True)

    # Load and preprocess data
    print("Loading data...")
    meta = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'meta_selected.csv'), index_col=0)
    matrix = pd.read_csv(os.path.join(DEMO_DATA_DIR, 'expr_selected.csv'), index_col=0).T

    # Select differential genes (DEGs) - same as in 01_0_Train_and_Interpret_WEDGE.py
    matrix_Degene_sub = DEgene_selected(matrix, path=PROJECT_ROOT)
    matrix = matrix_Degene_sub

    # Filter for HPV-related and NHPV samples
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]

    # Split data into train and test
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)

    # Encode labels: HPV_related = 0, NHPV = 1
    encoder = LabelEncoder()
    label_list = encoder.fit_transform(meta_train.CancerType.values)

    print(f"Matrix train shape: {matrix_train.shape}")
    print(f"Label distribution: HPV_related (0): {sum(label_list == 0)}, NHPV (1): {sum(label_list == 1)}")
    print(f"Using differential genes: {matrix_train.shape[1]} genes")

    # Generate augmented data for label 0 (HPV_related)
    print("\n" + "=" * 80)
    print("Generating augmented data for label 0 (HPV_related)...")
    print("=" * 80)
    generate_data(matrix_train, label_list, label=LABEL_HPV, savepath=AUG_DATA_DIR, **kwargs)

    # Generate augmented data for label 1 (NHPV)
    print("\n" + "=" * 80)
    print("Generating augmented data for label 1 (NHPV)...")
    print("=" * 80)
    generate_data(matrix_train, label_list, label=LABEL_NHPV, savepath=AUG_DATA_DIR, **kwargs)

    print("\n" + "=" * 80)
    print("Data augmentation complete!")
    print(f"Generated data saved to: {AUG_DATA_DIR}")
    print("=" * 80)
