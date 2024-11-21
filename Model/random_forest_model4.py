import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import logging
from typing import Dict, Tuple, List

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='random_forest_training.log'
)

# Model Constants and Hyperparameters
MODEL_CONFIG = {
    'n_channels': 60,          # Number of reduced channels
    'n_trees': 100,           # Number of trees in Random Forest
    'k_folds': 5,            # Number of cross-validation folds
    'early_stop_patience': 3, # Early stopping patience
    'min_improvement': 0.01,  # Minimum improvement threshold
    'feature_importance_threshold': 0.02,  # Feature selection threshold
    'train_size': 0.8,       # Training data split ratio
    'random_state': 42,      # Random seed for reproducibility
    'batch_size': 128,       # Batch size for processing
    'dataset_path': f'../Dataset/dataset10k_reduced_60.json'  # Added dataset path
}

# Optimized Random Forest Parameters
RF_PARAMS = {
    'n_estimators': MODEL_CONFIG['n_trees'],
    'max_depth': 8,          # Controlled tree depth
    'min_samples_split': 20, # Minimum samples for splitting
    'min_samples_leaf': 15,  # Minimum samples in leaf
    'max_features': 'sqrt',  # Feature selection strategy
    'random_state': MODEL_CONFIG['random_state'],
    'bootstrap': True,
    'max_samples': 0.7,      # Subsample size
    'class_weight': 'balanced_subsample',
    'n_jobs': -1,           # Parallel processing
    'oob_score': True,      # Out-of-bag score
    'warm_start': True      # Incremental learning
}

class IPARC:
    """
    Handles data preprocessing with:
    - Feature normalization
    - Kernel extraction
    - Batch processing
    - Memory-efficient data handling
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def prepare_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Optimized batch preparation"""
        batch = self.df.iloc[indices]
        
        # Vectorized feature extraction
        input_features = np.vstack([np.array(x).reshape(-1) for x in batch['input_reduced']])
        output_features = np.vstack([np.array(x).reshape(-1) for x in batch['output_reduced']])
        
        # Optimized normalization
        input_features = (input_features - np.mean(input_features, axis=1, keepdims=True)) / \
                        (np.std(input_features, axis=1, keepdims=True) + 1e-8)
        output_features = (output_features - np.mean(output_features, axis=1, keepdims=True)) / \
                         (np.std(output_features, axis=1, keepdims=True) + 1e-8)
        
        features = np.hstack([input_features, output_features])
        
        # Vectorized kernel extraction
        kernels = np.array([self._extract_kernel(k) for k in batch['kernel']])
        
        return {
            'features': features,
            'kernels': kernels
        }

    @staticmethod
    def _extract_kernel(kernel) -> int:
        """
        Extracts kernel index from kernel data, handling different formats.
        Args:
            kernel: Can be a single value, list, or numpy array
        Returns:
            int: The kernel value
        """
        # If kernel is already a single number
        if isinstance(kernel, (int, float, np.integer, np.floating)):
            return int(kernel)
        
        # Convert to numpy array if it's a list
        kernel_array = np.array(kernel)
        
        # If it's a nested array/list, flatten it
        kernel_array = kernel_array.flatten()
        
        # If it's one-hot encoded
        if set(np.unique(kernel_array)).issubset({0, 1}):
            try:
                return int(np.where(kernel_array == 1)[0][0])
            except IndexError:
                logging.error(f"Invalid one-hot encoded kernel: {kernel}")
                return 0
        
        # If it's a direct value
        if len(kernel_array) == 1:
            return int(kernel_array[0])
        
        logging.error(f"Unexpected kernel format: {kernel}")
        return 0


def analyze_feature_importance(model: RandomForestClassifier):
    """Analyzes and plots the feature importance of the given RandomForest model."""
    feature_importances = model.feature_importances_
    
    # Sort the feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
    plt.xticks(range(len(feature_importances)), indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()
    
    # Log the top 10 most important features
    top_features = indices[:10]
    for i in top_features:
        logging.info(f"Feature {i} Importance: {feature_importances[i]}")

def train_random_forest(
    train_dataset: IPARC, 
    test_dataset: IPARC, 
    n_estimators: int = MODEL_CONFIG['n_trees']
) -> Tuple[Dict[str, float], RandomForestClassifier]:
    """
    Training function with progress tracking and detailed logging
    """
    logging.info("Starting Random Forest training...")
    
    # Prepare data with progress bar
    with tqdm(total=2, desc="Data Preparation") as pbar:
        train_data = train_dataset.prepare_batch(range(len(train_dataset)))
        pbar.update(1)
        test_data = test_dataset.prepare_batch(range(len(test_dataset)))
        pbar.update(1)
    
    # Feature selection
    with tqdm(total=3, desc="Feature Selection") as pbar:
        initial_model = RandomForestClassifier(**RF_PARAMS)
        initial_model.fit(train_data['features'], train_data['kernels'])
        pbar.update(1)
        
        importances = initial_model.feature_importances_
        selected_features = importances > MODEL_CONFIG['feature_importance_threshold']
        pbar.update(1)
        
        if not np.any(selected_features):
            n_features = max(int(0.1 * len(importances)), 1)
            selected_features = np.argsort(importances)[-n_features:]
        pbar.update(1)
    
    # Model training with progress
    with tqdm(total=4, desc="Model Training") as pbar:
        X_train = train_data['features'][:, selected_features]
        X_test = test_data['features'][:, selected_features]
        pbar.update(1)
        
        final_model = RandomForestClassifier(**RF_PARAMS)
        final_model.fit(X_train, train_data['kernels'])
        pbar.update(1)
        
        train_pred = final_model.predict(X_train)
        test_pred = final_model.predict(X_test)
        pbar.update(1)
        
        metrics = {
            'train_acc': accuracy_score(train_data['kernels'], train_pred),
            'test_acc': accuracy_score(test_data['kernels'], test_pred),
            'oob_score': final_model.oob_score_ if hasattr(final_model, 'oob_score_') else None
        }
        pbar.update(1)
    
    # Log results
    logging.info(f"Training Accuracy: {metrics['train_acc']:.4f}")
    logging.info(f"Testing Accuracy: {metrics['test_acc']:.4f}")
    logging.info(f"OOB Score: {metrics['oob_score']:.4f}")
    
    return metrics, final_model

def k_fold_cross_validation(df: pd.DataFrame, k: int, n_estimators: int) -> Tuple[Dict, List]:
    """Optimized k-fold cross-validation with stratification"""
    from sklearn.model_selection import StratifiedKFold
    
    # Extract kernels for stratification
    kernels = df['kernel'].apply(lambda x: IPARC._extract_kernel(x))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=MODEL_CONFIG['random_state'])
    
    metrics_dict = {'train_acc': [], 'test_acc': [], 'oob_score': []}
    models = []
    
    best_score = float('-inf')
    patience_counter = 0
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, kernels)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_dataset, test_dataset = IPARC(train_df), IPARC(test_df)
        fold_metrics, model = train_random_forest(train_dataset, test_dataset, n_estimators)
        
        # Early stopping check
        if fold_metrics['test_acc'] > (best_score + MODEL_CONFIG['min_improvement']):
            best_score = fold_metrics['test_acc']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= MODEL_CONFIG['early_stop_patience']:
            logging.info(f"Early stopping triggered at fold {fold_idx + 1}")
            break
        
        # Store metrics
        for key in fold_metrics:
            if fold_metrics[key] is not None:
                metrics_dict[key].append(fold_metrics[key])
        
        models.append(model)
    
    return metrics_dict, models

def plot_metrics(metrics: Dict):
    """Enhanced visualization with confidence intervals"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for metric in ['train_acc', 'test_acc']:
        values = metrics[metric]
        mean = np.mean(values)
        std = np.std(values)
        
        x = range(len(values))
        ax.plot(x, values, label=f'{metric} (μ={mean:.3f}±{std:.3f})')
        ax.fill_between(x, 
                       np.array(values) - std,
                       np.array(values) + std,
                       alpha=0.1)
    
    ax.set_title('Model Performance Across Folds')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    try:
        df = pd.read_json(MODEL_CONFIG['dataset_path'])
        logging.info(f"Successfully loaded dataset from {MODEL_CONFIG['dataset_path']}")
        logging.info(f"Dataset shape: {df.shape}")
        
        metrics, models = k_fold_cross_validation(
            df, 
            k=MODEL_CONFIG['k_folds'], 
            n_estimators=MODEL_CONFIG['n_trees']
        )
        plot_metrics(metrics)
        
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {MODEL_CONFIG['dataset_path']}")
        raise
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise
