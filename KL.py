import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

# Load your dataset A.csv
data_A = pd.read_csv('openSmile_resampled/overall.csv')

# Extract features from your dataset (excluding labels if present)
features_A = data_A.drop(['label'], axis=1) if 'label' in data_A.columns else data_A


# Define a function to calculate KL divergence between two datasets
def kl_divergence(p, q):
    # Ensure p and q have the same shape
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]

    return entropy(p, q)


# Initialize lists to store the calculated KL divergences and p-values
kl_divergences = []
p_values = []

# Perform the non-parametric test by repeatedly splitting the dataset and calculating KL
num_splits = 10
original_kl = None  # Original KL Divergence

for _ in range(num_splits):
    # Randomly split the dataset into training and testing (80% training, 20% testing)
    train_data, test_data = train_test_split(features_A, test_size=0.2,
                                             random_state=None)  # Adjust random_state as needed

    if original_kl is None:
        original_kl = kl_divergence(train_data, test_data)

    # Calculate KL divergence between the training and testing datasets
    kl_value = kl_divergence(train_data, test_data)
    kl_divergences.append(kl_value)

    # Calculate the p-value
    p_value = np.mean(np.array(kl_divergences) >= original_kl)
    p_values.append(p_value)

# Calculate the standard deviation of p-values
p_values_std = np.std(p_values)

# Print the results
print(f"Original KL Divergence: {original_kl}")
print(f"Mean p-value: {np.mean(p_values)}")
print(f"Standard Deviation of p-values: {p_values_std}")
