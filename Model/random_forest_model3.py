import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
n_c = 10  # Number of reduced channels
n_trees = 500  # Number of trees in the random forest
k_folds = 5  # Number of K-folds
file = f'../Dataset/dataset5k_reduced_{n_c}.json'
# file = f'../Dataset_Recentlygenerated/dataset1k_reduced_{n_c}.json'  # Alternative dataset path


class IPARC:
    """Dataset class for handling and processing batch data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def prepare_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Prepares a batch of features, operations, and kernels."""
        batch = self.df.iloc[indices]

        # Extract and normalize features
        features_list = []
        for _, row in batch.iterrows():
            input_features = np.array(row['input_reduced']).reshape(-1)
            output_features = np.array(row['output_reduced']).reshape(-1)
            
            # Normalize input and output features
            input_features = (input_features - np.mean(input_features)) / (np.std(input_features) + 1e-8)
            output_features = (output_features - np.mean(output_features)) / (np.std(output_features) + 1e-8)
            
            features = np.concatenate([input_features, output_features])
            features_list.append(features)
        
        features = np.array(features_list)

        # Extract operations and kernels
        operations = []
        kernels = []
        
        for _, row in batch.iterrows():
            # Extract operation
            op = row['operation']
            if isinstance(op, (list, np.ndarray)):
                op_value = op[0][0] if isinstance(op[0], (list, np.ndarray)) else op[0]
            else:
                op_value = op
            operations.append(op_value)
            
            # Extract kernel
            kernel = self._extract_kernel(row['kernel'])
            kernels.append(kernel)
            
        # Log distribution information
        unique_kernels = np.unique(kernels)
        kernel_counts = np.bincount(kernels)
        logging.info(f"Unique kernel values: {unique_kernels}")
        logging.info(f"Kernel value counts: {kernel_counts}")
        
        # Check for potential issues
        if len(unique_kernels) == 1:
            logging.warning("Only one unique kernel value found! This will result in 100% accuracy.")
        
        if np.any(kernel_counts == 0):
            logging.warning("Some kernel classes have zero samples!")

        return {
            'features': features,
            'operations': np.array(operations),
            'kernels': np.array(kernels)
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
    train_dataset: IPARC, test_dataset: IPARC, n_estimators: int = 100
) -> Tuple[Dict[str, Dict[str, float]], Tuple[RandomForestClassifier, RandomForestClassifier]]:
    """Trains random forest models and evaluates metrics."""
    # Prepare data
    train_data = train_dataset.prepare_batch(range(len(train_dataset)))
    test_data = test_dataset.prepare_batch(range(len(test_dataset)))

    # Log data distribution
    logging.info(f"Training set size: {len(train_data['kernels'])}")
    logging.info(f"Test set size: {len(test_data['kernels'])}")
    
    # Modified model parameters to prevent overfitting
    model_params = {
        'n_estimators': n_estimators,
        'max_depth': 10,  # Reduced from 15 to prevent overfitting
        'min_samples_split': 15,  # Increased from 5
        'min_samples_leaf': 10,  # Increased from 3
        'max_features': 'sqrt',
        'random_state': 42,
        'bootstrap': True,
        'max_samples': 0.8,  # Use a smaller subset of samples per tree to improve generalization
        'class_weight': 'balanced',  # Balancing the class weights to avoid bias towards dominant class
        'n_jobs': -1,
        'oob_score': True  # Add out-of-bag score to monitor overfitting
    }
    
    operation_model = RandomForestClassifier(**model_params)
    kernel_model = RandomForestClassifier(**model_params)

    # Train models
    operation_model.fit(train_data['features'], train_data['operations'])
    kernel_model.fit(train_data['features'], train_data['kernels'])

    # Get predictions
    train_kernel_pred = kernel_model.predict(train_data['features'])
    test_kernel_pred = kernel_model.predict(test_data['features'])
    
    # Log prediction distributions
    logging.info("Prediction distribution:")
    train_pred_dist = np.bincount(train_kernel_pred)
    test_pred_dist = np.bincount(test_kernel_pred)
    
    for kernel_idx, (train_count, test_count) in enumerate(zip(train_pred_dist, test_pred_dist)):
        if train_count > 0 or test_count > 0:
            logging.info(f"Kernel {kernel_idx}:")
            logging.info(f"  Training: {train_count} predictions")
            logging.info(f"  Testing: {test_count} predictions")

    # Calculate metrics
    metrics = {
        'train': {
            'operation': accuracy_score(train_data['operations'], 
                                     operation_model.predict(train_data['features'])),
            'kernel': accuracy_score(train_data['kernels'], 
                                   train_kernel_pred)
        },
        'test': {
            'operation': accuracy_score(test_data['operations'], 
                                     operation_model.predict(test_data['features'])),
            'kernel': accuracy_score(test_data['kernels'], 
                                   test_kernel_pred)
        }
    }

    # After training the models
    logging.info("\nKernel model feature importance:")
    analyze_feature_importance(kernel_model)

    return metrics, (operation_model, kernel_model)


def k_fold_cross_validation(df: pd.DataFrame, k: int, n_estimators: int) -> Tuple[Dict, List]:
    """Performs K-fold cross-validation with early stopping."""
    metrics_dict = {'train': {'operation': [], 'kernel': []}, 
                   'test': {'operation': [], 'kernel': []},
                   'oob': {'operation': [], 'kernel': []}}  # Out-of-bag scores
    models = []

    window = len(df) // k
    logging.info(f"Dataset size: {len(df)}, Fold size: {window}")

    for i in tqdm(range(k), desc="Cross Validation Progress"):
        start, end = i * window, (i + 1) * window
        train_df = pd.concat([df.iloc[:start], df.iloc[end:]])
        test_df = df.iloc[start:end]

        train_dataset, test_dataset = IPARC(train_df), IPARC(test_df)
        fold_metrics, fold_models = train_random_forest(train_dataset, test_dataset, n_estimators)
        
        # Early stopping check based on OOB score
        operation_model, kernel_model = fold_models
        if hasattr(kernel_model, 'oob_score_'):
            metrics_dict['oob']['kernel'].append(kernel_model.oob_score_)
        
        # Store metrics
        for key in fold_metrics['train']:
            metrics_dict['train'][key].append(fold_metrics['train'][key])
            metrics_dict['test'][key].append(fold_metrics['test'][key])

        models.append(fold_models)

    return metrics_dict, models


def plot_metrics(metrics: Dict):
    """Plot metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot train and test accuracies for operation and kernel
    for metric in ['kernel']:
        ax.plot(metrics['train'][metric], label=f'{metric} Train Accuracy', color='black', marker='o')
        ax.plot(metrics['test'][metric], label=f'{metric} Test Accuracy', color='red', marker='x')

    # Set plot labels and title
    ax.set_title('Train and Test Accuracy')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.legend()

    plt.tight_layout()
    plt.show()



# Main script
if __name__ == "__main__":
    df = pd.read_json(file)
    metrics, _ = k_fold_cross_validation(df, k=k_folds, n_estimators=n_trees)

    # Plot results
    plot_metrics(metrics)
