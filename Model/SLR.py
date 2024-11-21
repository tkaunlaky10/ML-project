import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics import Accuracy

from matplotlib import pyplot as plt
from tqdm import tqdm
n_c = 60
hidden_size = 16
file = f'../Dataset/dataset1k_reduced_{n_c}.json'
### Datasets

# dataloaders
class IPARC(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'img_in': torch.tensor(row['input_reduced']).reshape(-1),
            'img_out': torch.tensor(row['output_reduced']).reshape(-1),
            'operation': torch.tensor(row['operation']).reshape(1, -1),
            'kernel': torch.tensor(row['kernel']),
        }

    @staticmethod
    def collate(batch):
        return {
            'img_in': torch.stack([x['img_in'] for x in batch]),
            'img_out': torch.stack([x['img_out'] for x in batch]),
            'operation': torch.stack([x['operation'] for x in batch]),
            'kernel': torch.stack([x['kernel'] for x in batch]),
        }
### Model

class Model(nn.Module):
    def __init__(self, n_c, hdsz):
        super(Model, self).__init__()
        self.op_linreg = nn.Linear(4 * (n_c + n_c) + hdsz, 1)
        self.kernel_linreg = nn.Linear(4 * (n_c + n_c) + hdsz, 8)
        self.hidden_encoder = nn.Linear(hdsz + 1 + 8, hdsz)

    def forward(self, img_in, img_out, op_prev, kernel_prev, hidden):
        hidden = torch.tanh(self.hidden_encoder(torch.cat([hidden, op_prev, kernel_prev], dim=-1)))

        cat = torch.cat([img_in, img_out, hidden], dim=-1)
        op_logit = self.op_linreg(cat).squeeze(-1)
        kernel_logit = self.kernel_linreg(cat)

        return op_logit, kernel_logit, hidden
### Training Loop

# trainig loop
def train(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    acc_bin = Accuracy(task='binary').to(device)
    acc_multi = Accuracy(task='multiclass', num_classes=8).to(device)

    losses = {'train': [], 'test': []}
    metrics = {'train': {'operation': [],'kernel': [], }, 'test': { 'operation': [], 'kernel': [] }}

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        loss_tot, op_acc_tot, kernel_acc_tot = 0, 0, 0
        for batch in train_loader:
            bs = batch['img_in'].shape[0]

            img_in = batch['img_in'].to(device)
            img_out = batch['img_out'].to(device)
            op = batch['operation'].to(device)
            kernel = batch['kernel'].to(device)

            prev_op, prev_kernel = torch.zeros_like(op[:, :, 0], device=device), torch.zeros_like(
                kernel[:, 0], device=device
            )
            hidden = torch.zeros(bs, hidden_size, device=device)

            loss, op_acc, kernel_acc = 0, 0, 0
            n_seq = op.shape[2]
            for i in range(n_seq):
                op_logit, kernel_logit, hidden = model(
                    img_in, img_out, prev_op, prev_kernel, hidden
                )
                op_loss = criterion_bce(op_logit, op[:, 0, i].float())
                kernel_loss = criterion_ce(kernel_logit, ((kernel[:, i] == 1).nonzero(as_tuple=True)[1]).long())
                loss += op_loss + kernel_loss

                op_acc += acc_bin(op_logit, op[:, 0, i].float())
                kernel_acc += acc_multi(kernel_logit, ((kernel[:, i] == 1).nonzero(as_tuple=True)[1]).long())
                
                prev_op = op[:, :, i]
                prev_kernel = kernel[:, i]

            loss /= n_seq
            loss_tot += loss.detach().cpu().item()
            
            op_acc /= n_seq
            op_acc_tot += op_acc.detach().cpu().item()
            
            kernel_acc /= n_seq
            kernel_acc_tot += kernel_acc.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses['train'].append(loss_tot / len(train_loader))
        metrics['train']['operation'].append(op_acc_tot / len(train_loader))
        metrics['train']['kernel'].append(kernel_acc_tot / len(train_loader))

        model.eval()
        loss_tot, op_acc_tot, kernel_acc_tot = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                bs = batch['img_in'].shape[0]

                img_in = batch['img_in'].to(device)
                img_out = batch['img_out'].to(device)
                op = batch['operation'].to(device)
                kernel = batch['kernel'].to(device)

                prev_op, prev_kernel = torch.zeros_like(op[:, :, 0], device=device), torch.zeros_like(
                    kernel[:, 0], device=device
                )
                hidden = torch.zeros(bs, hidden_size, device=device)

                loss, op_acc, kernel_acc = 0, 0, 0
                n_seq = op.shape[2]
                for i in range(n_seq):
                    op_logit, kernel_logit, hidden = model(
                        img_in, img_out, prev_op, prev_kernel, hidden
                    )
                    op_loss = criterion_bce(op_logit, op[:, 0, i].float())
                    kernel_loss = criterion_ce(kernel_logit, ((kernel[:, i] == 1).nonzero(as_tuple=True)[1]).long())
                    loss += op_loss + kernel_loss

                    op_acc += acc_bin(op_logit, op[:, 0, i].float())
                    kernel_acc += acc_multi(kernel_logit, ((kernel[:, i] == 1).nonzero(as_tuple=True)[1]).long())
                    
                    prev_op = op[:, :, i]
                    prev_kernel = kernel[:, i]

                loss /= n_seq
                loss_tot += loss.detach().cpu().item()
                
                op_acc /= n_seq
                op_acc_tot += op_acc.detach().cpu().item()
                
                kernel_acc /= n_seq
                kernel_acc_tot += kernel_acc.detach().cpu().item()
            losses['test'].append(loss_tot / len(test_loader))
            metrics['test']['operation'].append(op_acc_tot / len(test_loader))
            metrics['test']['kernel'].append(kernel_acc_tot / len(test_loader))

        pbar.set_description(f'{epoch + 1} | tr-loss: {losses["train"][-1]:.4f} | tr-op: {metrics["train"]["operation"][-1]:.4f} | te-op: {metrics["test"]["operation"][-1]:.4f} | tr-ker: {metrics["train"]["kernel"][-1]:.4f} | te-ker: {metrics["test"]["kernel"][-1]:.4f}')

    return losses, metrics
### K-Fold Cross Validation

# Load and check the data
df = pd.read_json(file)
print("Initial dataset size:", len(df))
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
print("After shuffle dataset size:", len(df))
def k_fold(df, n_c, hidden_size, epochs=10, lr=0.01, k=1, idx=0, device='cpu'):
    # Print dataset size for debugging
    print(f"Total dataset size: {len(df)}")
    
    # Calculate window size
    window = len(df) // k
    print(f"Window size: {window}")
    
    # Calculate indices
    start_idx = idx * window
    end_idx = (idx + 1) * window
    print(f"Start idx: {start_idx}, End idx: {end_idx}")
    
    # Split data using iloc
    test_df = df.iloc[start_idx:end_idx].copy()
    train_df = pd.concat([
        df.iloc[:start_idx], 
        df.iloc[end_idx:]
    ]).copy()
    
    # Print sizes for debugging
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Dataset split resulted in empty training or test set!")

    # Create data loaders
    train_loader = DataLoader(
        IPARC(train_df), 
        batch_size=min(128, len(train_df)), 
        shuffle=True, 
        collate_fn=IPARC.collate
    )
    test_loader = DataLoader(
        IPARC(test_df), 
        batch_size=min(128, len(test_df)), 
        shuffle=False, 
        collate_fn=IPARC.collate
    )

    model = Model(n_c, hidden_size)
    losses, metrics = train(model, train_loader, test_loader, epochs, lr, device)

    return losses, metrics

# Main training loop
n_epochs = 30
k = 5  # Use 5-fold cross validation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and check data
print(f"Loading data from: {file}")
df = pd.read_json(file)
print(f"Initial dataset size: {len(df)}")
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
print(f"After shuffle dataset size: {len(df)}")

loss_dict = {'train': np.zeros(n_epochs), 'test': np.zeros(n_epochs)}
metrics_dict = {
    'train': {'operation': np.zeros(n_epochs), 'kernel': np.zeros(n_epochs)}, 
    'test': {'operation': np.zeros(n_epochs), 'kernel': np.zeros(n_epochs)}
}

# Run k-fold cross validation
for i in range(k):  # Changed from len(df) // k to k
    print(f"\nFold {i+1}/{k}")
    losses, metrics = k_fold(df, n_c, hidden_size, epochs=n_epochs, lr=0.01, k=k, idx=i, device=device)
    
    loss_dict['train'] += np.array(losses['train'])
    loss_dict['test'] += np.array(losses['test'])
    metrics_dict['train']['operation'] += np.array(metrics['train']['operation'])
    metrics_dict['train']['kernel'] += np.array(metrics['train']['kernel'])
    metrics_dict['test']['operation'] += np.array(metrics['test']['operation'])
    metrics_dict['test']['kernel'] += np.array(metrics['test']['kernel'])

# Average the results
loss_dict['train'] /= k
loss_dict['test'] /= k
metrics_dict['train']['operation'] /= k
metrics_dict['train']['kernel'] /= k
metrics_dict['test']['operation'] /= k
metrics_dict['test']['kernel'] /= k

plt.plot(loss_dict['train'], label='train')
plt.plot(loss_dict['test'], label='test')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(metrics_dict['train']['operation'], label='train')
plt.plot(metrics_dict['test']['operation'], label='test')
plt.legend()
plt.title('Operation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Operation Accuracy')
plt.show()

plt.plot(metrics_dict['train']['kernel'], label='train', color='black')
plt.plot(metrics_dict['test']['kernel'], label='test', color='red')
plt.legend()
plt.title('Accuracy (in%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in%)')
plt.show()
