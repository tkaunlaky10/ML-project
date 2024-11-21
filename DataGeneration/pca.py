import pandas as pd
import ast
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import TruncatedSVD, PCA
df = pd.read_json('../Dataset_Recentlygenerated/dataset.json')
df.head(5)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:1000]
def pca_explnation(features, n_c):
    pca = PCA(n_components=n_c, random_state=42)
    pca.fit(features)
    return pca.explained_variance_ratio_.sum()

def pca_transform(features, n_c):
    pca = PCA(n_components=n_c, random_state=42)
    pca.fit(features)
    return pca.transform(features)
inps = np.concatenate([np.array(i) for i in df['input'].values], axis=0)
outs = np.concatenate([np.array(i) for i in df['output'].values], axis=0)
imgs = np.concatenate([inps, outs], axis=0)
print(imgs.shape)

n, h, w = imgs.shape
imgs_flat = imgs.reshape(n, h*w)
# Plot explained variance ratio vs number of components
# x = range(1, 225)
# y = [pca_explnation(imgs_flat, i) for i in x]
# This code creates a plot to visualize how much variance is explained by different numbers of PCA components
# x-axis: number of components (1-224)
# y-axis: cumulative explained variance ratio for each number of components

# plt.figure(figsize=(10, 5))
# plt.plot(x, y)
# plt.show()
# Creates and displays a line plot showing how adding more PCA components increases the explained variance
# Helps determine optimal number of components to use

# Test reconstruction with 10 components
# pca = PCA(n_components=10, random_state=42)
# pca.fit(imgs_flat)
# Initializes PCA with 10 components and fits it to the flattened image data
# random_state=42 ensures reproducibility

# reduced = pca.transform(imgs_flat)
# Transforms the original data into the lower-dimensional space (10 components)
# This is the compressed representation of the images

# inverse = pca.inverse_transform(reduced)
# inverse = inverse.reshape(n, h, w)
# Reconstructs the images from the compressed representation
# Reshapes the data back to original image dimensions (height x width)

# plt.imshow(imgs[0].reshape(h, w))
# plt.show()
# Displays the original first image for comparison

# plt.imshow(inverse[0] >= 0.5)
# plt.show()
# Displays the reconstructed first image after PCA compression/decompression
# >= 0.5 thresholds the values to create a binary image
# This allows visual comparison of reconstruction quality
n_c = 60
reduced = pca_transform(imgs_flat, n_c)
print(reduced.shape)
reduced_inps = reduced[:len(inps)].tolist()
reduced_inps = [reduced_inps[i:i+4] for i in range(0, len(reduced_inps), 4)]
reduced_outs = reduced[len(inps):]
reduced_outs = [reduced_outs[i:i+4] for i in range(0, len(reduced_outs), 4)]

df_reduced = df.copy()
df_reduced['input_reduced'] = reduced_inps
df_reduced['output_reduced'] = reduced_outs

print(np.array(df_reduced['input_reduced'][0]).shape)
df_reduced.head(2)
df_reduced.to_json(f'../Dataset_Recentlygenerated/dataset1k_reduced_{n_c}.json')