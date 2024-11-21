# Image Processing Algorithm Recognition Challenge (IPARC)

This project compares Sequential Logistic Regression (SLR), Random Forest (RF), and Decision Tree models on the IPARC CatA Dataset for predicting image transformation sequences.

## Research Context

### Problem Statement
IPARC's CatA Simple task consists of operations on grayscale images. Each task involves:
- A sequence of 4 dilation operations followed by 4 erosion operations
- 3x3 matrices as structuring elements
- 4 input-output image pairs per example
- Output images generated by applying operations on input images

**Research Question:** Given the input, output, and a prefix of the associated transformation sequence, can a sequentially applied logistic regression model perform better than a random forest at predicting the succeeding transformation of the sequence?

### Hypothesis
Sequential Logistic Regression may outperform Random Forest and Decision Trees on partial datasets reduced by Principal Component Analysis, particularly for simpler reasoning tasks. This advantage stems from:
- Efficiency in handling reduced-dimension data
- Better interpretability
- Suitability for datasets that intentionally omit irrelevant information

### Significance
Partial datasets are frequently encountered in machine learning applications due to:
- Limitations in data collection
- Preprocessing challenges
- Need for progressive decision-making
- Real-world data availability constraints

Understanding model performance on such subsets is essential for determining:
- Robustness
- Scalability
- Generalizability in real-world scenarios

## Project Structure

```
project-root/
├── Model/
│   ├── SLR.py                    # Sequential Logistic Regression
│   ├── random_forest_model4.py   # Random Forest with k-fold CV
│   └── decision_tree_model.py    # Decision Tree with metrics
│
├── Dataset/                      # Original dataset
│   ├── CatA_Simple/             # Original task JSON files
│   ├── dataset.json             # Combined dataset
│   ├── dataset1k_reduced_60.json # Reduced dimension dataset
│   └── data-to-json.py          # Data conversion utility
│
├── Dataset_Recentlygenerated/    # Generated datasets
│   ├── CatA_Simple/             # Generated task files
│   ├── dataset.json             # Combined generated dataset
│   └── dataset*k_reduced_*.json  # Reduced dimension datasets
│
└── DataGeneration/
    ├── datagen.py               # Dataset generation script
    └── pca.py                   # Dimensionality reduction
```

## Data Management

### Dataset Organization

The project maintains two primary dataset directories:

1. `Dataset/`: Original IPARC dataset
   - Contains the baseline CatA_Simple tasks
   - Includes pre-processed and dimensionally reduced versions
   - Used for model benchmarking and validation

2. `Dataset_Recentlygenerated/`: Generated datasets
   - Created using datagen.py with customizable parameters
   - Maintains same structure as original dataset
   - Used for experimentation and model training
  

### Data Generation Process

1. Generate new datasets:
```bash
cd DataGeneration
python datagen.py
```

2. Convert to JSON format:
```bash
python Dataset/data-to-json.py
```

3. Apply dimensionality reduction:
```bash
python DataGeneration/pca.py
```

### Data Processing Pipeline

The data processing workflow includes:

1. Raw Data Generation:
   - Creates synthetic image transformation sequences
   - Configurable parameters for dataset size and complexity
   - Automatic directory creation with timestamps

2. Data Structuring:
   - Converts raw data to structured JSON format
   - Organizes input-output image pairs
   - Maintains transformation sequence metadata

3. Dimensionality Reduction:
   - PCA analysis for feature reduction
   - Configurable number of components
   - Generates visualization of data reconstruction

Example usage:
```python
# Load and process data
data = load_dataset("Dataset_Recentlygenerated/dataset.json")
visualize_image_pair(data[0]['input'], data[0]['output'])

# Apply dimensionality reduction
reduced_data = reduce_dimensions(data, n_components=60)
save_dataset(reduced_data, "dataset1k_reduced_60.json")
```

### Configuration

1. Data Generation Settings:
```python
# In datagen.py
dataset_dir = f"../Dataset_Recentlygenerated/CatA_Simple"
```

2. Model Training Paths:
```python
# In Model/slr.py or Model/rf.py
file = f'./Dataset_Recentlygenerated/dataset1k_reduced_{n_c}.json'
```

## Model Training and Evaluation

1. Train models:
```bash
python Model/slr.py        # Sequential Logistic Regression
python Model/rf.py         # Random Forest
python Model/decision_tree_model.py  # Decision Tree
```

2. Evaluation metrics include:
   - Accuracy on test set
   - K-fold cross-validation results
   - Prediction consistency
   - Model-specific metrics

## Requirements
   
    numpy>=1.19.2        # For numerical operations and array handling
    torch>=1.9.0         # For neural network operations
    matplotlib>=3.3.2    # For data visualization
    scikit-learn>=0.24.2 # For machine learning algorithms
    scipy>=1.6.0         # For scientific computations
    pandas>=1.2.0        # For data manipulation
    tqdm>=4.50.2         # For progress bars
    torchmetrics>=0.5.0  # For model metrics

### Python Version
- Python 3.8 or higher
- pip package manager

### Installation Options

1. Using pip (recommended):
```bash
pip install -r requirements.txt
```

2. Manual installation:
```bash
pip install numpy torch matplotlib scikit-learn scipy pandas tqdm torchmetrics pillow joblib seaborn
```

3. For GPU support (optional but recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### System Requirements
- Python 3.8+
- Sufficient RAM for dataset processing
- GPU optional but recommended for larger datasets

## Notes
- Ensure correct path configuration in all scripts
- Generated datasets are automatically timestamped
- Use reduced dimension datasets for faster training
- Monitor system resources when processing large datasets


