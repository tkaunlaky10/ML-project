# First cell - imports
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics import Accuracy

from sklearn.tree import DecisionTreeClassifier  # Add this
from sklearn.metrics import accuracy_score      # Add this

from matplotlib import pyplot as plt
from tqdm import tqdm
# Constants
n_c = 60
n_epochs = 30  # For consistency in plotting
file = f'../Dataset/dataset5k_reduced_{n_c}.json'
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
    
    def prepare_for_tree(self):
        """Prepare data for decision tree training"""
        features = []
        operations = []
        kernels = []
        
        for i in range(len(self)):
            sample = self[i]
            # Concatenate input and output features
            feature = torch.cat([sample['img_in'], sample['img_out']]).numpy()
            features.append(feature)
            
            # Get operation and kernel labels
            operations.append(sample['operation'].numpy().flatten()[0])  # Get first operation
            kernels.append(sample['kernel'].numpy()[0])  # Get first kernel
            
        return np.array(features), np.array(operations), np.array(kernels)
    
    
def train_decision_trees(train_dataset, test_dataset, epoch_metrics=None):
    """Training function with epoch-wise metrics tracking"""
    X_train, y_operation_train, y_kernel_train = train_dataset.prepare_for_tree()
    X_test, y_operation_test, y_kernel_test = test_dataset.prepare_for_tree()
    
    # Train trees with different max_depths to simulate epochs
    metrics_history = {
        'train': {'operation': [], 'kernel': []},
        'test': {'operation': [], 'kernel': []}
    }
    
    # Train multiple trees with increasing depth
    for depth in range(1, n_epochs + 1):
        # Train operation tree
        operation_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        operation_tree.fit(X_train, y_operation_train)
        
        # Train kernel tree
        kernel_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        kernel_tree.fit(X_train, y_kernel_train)
        
        # Calculate metrics
        train_op_acc = accuracy_score(y_operation_train, operation_tree.predict(X_train))
        test_op_acc = accuracy_score(y_operation_test, operation_tree.predict(X_test))
        train_kernel_acc = accuracy_score(y_kernel_train, kernel_tree.predict(X_train))
        test_kernel_acc = accuracy_score(y_kernel_test, kernel_tree.predict(X_test))
        
        # Store metrics
        metrics_history['train']['operation'].append(train_op_acc)
        metrics_history['test']['operation'].append(test_op_acc)
        metrics_history['train']['kernel'].append(train_kernel_acc)
        metrics_history['test']['kernel'].append(test_kernel_acc)
    
    return metrics_history, (operation_tree, kernel_tree)
def k_fold(df, k=5, idx=0):
    """K-fold cross validation with metrics tracking"""
    print(f"Total dataset size: {len(df)}")
    window = len(df) // k
    
    start_idx = idx * window
    end_idx = (idx + 1) * window
    
    test_df = df.iloc[start_idx:end_idx].copy()
    train_df = pd.concat([df.iloc[:start_idx], df.iloc[end_idx:]]).copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    train_dataset = IPARC(train_df)
    test_dataset = IPARC(test_df)
    
    metrics_history, models = train_decision_trees(train_dataset, test_dataset)
    return metrics_history, models

# Main training loop
print(f"Loading data from: {file}")
df = pd.read_json(file)
print(f"Initial dataset size: {len(df)}")
df = df.sample(frac=1).reset_index(drop=True)
print(f"After shuffle dataset size: {len(df)}")

# Initialize metrics tracking
metrics_dict = {
    'train': {'operation': np.zeros(n_epochs), 'kernel': np.zeros(n_epochs)},
    'test': {'operation': np.zeros(n_epochs), 'kernel': np.zeros(n_epochs)}
}

# Run k-fold cross validation
k = 5
for i in tqdm(range(k)):
    print(f"\nFold {i+1}/{k}")
    fold_metrics, _ = k_fold(df, k=k, idx=i)
    
    # Accumulate metrics
    for phase in ['train', 'test']:
        for metric in ['operation', 'kernel']:
            metrics_dict[phase][metric] += np.array(fold_metrics[phase][metric])

# Average the results
for phase in ['train', 'test']:
    for metric in ['operation', 'kernel']:
        metrics_dict[phase][metric] /= k

# Plot results
plt.figure(figsize=(15, 5))

# Plot Operation Accuracy over tree depth
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs + 1), metrics_dict['train']['operation'], 
         label='Train', marker='o', markersize=4)
plt.plot(range(1, n_epochs + 1), metrics_dict['test']['operation'], 
         label='Test', marker='o', markersize=4)
plt.legend()
plt.title('Operation Accuracy vs Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Operation Accuracy')
plt.grid(True)

# Plot Kernel Accuracy over tree depth
plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs + 1), metrics_dict['train']['kernel'], 
         label='Train', marker='o', markersize=4)
plt.plot(range(1, n_epochs + 1), metrics_dict['test']['kernel'], 
         label='Test', marker='o', markersize=4)
plt.legend()
plt.title('Kernel Accuracy vs Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Kernel Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final metrics
print("\nFinal Average Metrics:")
print(f"Train Operation Accuracy: {metrics_dict['train']['operation'][-1]:.4f}")
print(f"Test Operation Accuracy: {metrics_dict['test']['operation'][-1]:.4f}")
print(f"Train Kernel Accuracy: {metrics_dict['train']['kernel'][-1]:.4f}")
print(f"Test Kernel Accuracy: {metrics_dict['test']['kernel'][-1]:.4f}")

# Plot Feature Importance
plt.figure(figsize=(15, 5))

# Operation Feature Importance
plt.subplot(1, 2, 1)
operation_importance = operation_tree.feature_importances_
plt.bar(range(len(operation_importance)), operation_importance)
plt.title('Operation Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.grid(True)

# Kernel Feature Importance
plt.subplot(1, 2, 2)
kernel_importance = kernel_tree.feature_importances_
plt.bar(range(len(kernel_importance)), kernel_importance)
plt.title('Kernel Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(15, 5))

# Operation Confusion Matrix
plt.subplot(1, 2, 1)
op_cm = confusion_matrix(y_operation_test, operation_tree.predict(X_test))
sns.heatmap(op_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Operation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Kernel Confusion Matrix
plt.subplot(1, 2, 2)
kernel_cm = confusion_matrix(y_kernel_test, kernel_tree.predict(X_test))
sns.heatmap(kernel_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Kernel Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()