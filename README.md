# Image Processing Algorithm Recognition Challenge (IPARC)

This project compares Sequential Logistic Regression (SLR), Random Forest (RF), and Decision Tree models on the IPARC CatA Dataset for predicting image transformation sequences.

## Overview

The project evaluates model performance on partial datasets of image transformations, focusing on:
- Sequential dilation and erosion operations
- 3x3 structuring elements
- Grayscale image processing
- Input-output image pair analysis

## Project Structure

```
project-root/
├── data/               # Dataset and data processing
│   ├── raw/           # Original IPARC datasets
│   └── processed/     # Processed image data
│
├── models/            # Model implementations
│   ├── slr.py        # Sequential Logistic Regression
│   ├── rf.py         # Random Forest model
│   └── dt.py         # Decision Tree model
```

## Data Configuration

### Generating New Data
The data generation script (`data/datagen.py`) creates datasets with configurable parameters:

```python
param = {
    'img_size': 15,           # Size of generated images
    'se_size': 5,            # Size of structuring elements
    'seq_length': 4,         # Length of transformation sequence
    'no_examples_per_task': 4, # Number of examples per task
    'no_colors': 3           # Number of colors used
}
```

### Data Processing (data.py)
The data processing script handles:
- Loading and preprocessing of generated datasets
- Visualization of input/output image pairs
- Data reduction and normalization
- Dataset analysis and statistics

Example usage:
```python
# Load and visualize data
data = load_dataset("path/to/dataset.json")
visualize_image_pair(data[0]['input'], data[0]['output'])

# Reduce dataset dimensions
reduced_data = reduce_dimensions(data, n_components=60)

# Save processed data
save_dataset(reduced_data, "processed_dataset.json")
```

### Data Paths
To change data paths:

1. For data generation:
```python
# In datagen.py
dataset_dir = f"../Dataset_{timestamp}/CatA_Simple"  # Change output directory
```

2. For model training:
```python
# In models/slr.py or models/rf.py
file = f'./Dataset/dataset1k_reduced_{n_c}.json'  # Change input dataset path
```

## Key Features

- Data Generation: Creates synthetic image transformation sequences
- Model Comparison: Evaluates SLR vs RF/DT performance
- Sequence Prediction: Predicts next transformation in a sequence
- Performance Metrics: Accuracy and consistency measurements

## Results

Our experiments show that Sequential Logistic Regression performs particularly well on:
- Smaller datasets and partial datasets
- Simple transformation sequences
- Progressive prediction tasks

## Usage

1. Generate dataset:
```bash
python data/datagen.py
```

2. Convert data to JSON format:
```bash
python Dataset/data-to-json.py
```
This script converts the generated dataset into a structured JSON format for further processing.

3. Apply PCA dimensionality reduction:
```bash
python Data/pca.py
```
This step:
- Reduces the dimensionality of the image data
- Analyzes variance explained by different numbers of components
- Creates visualizations of original vs. reconstructed images
- Generates reduced datasets (dataset1k_reduced_60.json, etc.)

4. Train and evaluate models:
```bash
python models/slr.py  # For Sequential Logistic Regression
python models/rf.py   # For Random Forest
```

Note: Make sure to run these scripts in sequence as each step depends on the output of the previous step.

### Data Processing Flow
1. Raw data generation (datagen.py)
2. Data structuring (data-to-json.py)
3. Dimensionality reduction (pca.py)
4. Model training and evaluation

Each script can be configured through parameters at the top of the file. See individual script documentation for details.

## Requirements

### Python Version
- Python 3.8+

### Core Dependencies
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn
- SciPy
- Pandas
- tqdm

### Additional Libraries
- torchmetrics (for accuracy metrics)
- datetime (for timestamp generation)
- json (for data handling)
- logging (for model logging)
- os, sys (for file and path handling)

### Installation
```bash
pip install numpy torch matplotlib scikit-learn scipy pandas tqdm torchmetrics
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```


