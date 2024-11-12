import pandas as pd
import numpy as np

# Entropy calculation
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Information Gain calculation
def information_gain(data, attribute, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[attribute], return_counts=True)

    # Calculate weighted entropy of subsets
    weighted_entropy = sum((counts[i] / np.sum(counts)) * entropy(data[data[attribute] == vals[i]][target_name])
                           for i in range(len(vals)))

    # Information Gain
    return total_entropy - weighted_entropy

# Split Information calculation
def split_information(data, attribute):
    vals, counts = np.unique(data[attribute], return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Gain Ratio calculation
def gain_ratio(data, attribute, target_name):
    ig = information_gain(data, attribute, target_name)
    si = split_information(data, attribute)
    return ig / si if si != 0 else 0

# Load dataset
file_path = input("Enter the path to your CSV dataset: ")
target_name = input("Enter the name of the target column: ")

data = pd.read_csv(file_path)

# Ensure the target column is present in the dataset
if target_name not in data.columns:
    raise ValueError(f"Column '{target_name}' not found in the dataset.")

# List of attributes excluding the target
attributes = [col for col in data.columns if col != target_name]

# Calculate Gain Ratio for each attribute and store in a DataFrame
gain_ratios = {attr: gain_ratio(data, attr, target_name) for attr in attributes}
gain_ratios_df = pd.DataFrame(list(gain_ratios.items()), columns=['Attribute', 'Gain Ratio'])

# Find the attribute with the maximum gain ratio
max_gain_ratio_attr = gain_ratios_df.loc[gain_ratios_df['Gain Ratio'].idxmax()]

# Display results
print("Gain Ratio for each attribute:")
print(gain_ratios_df)
print("\nAttribute with maximum gain ratio:")
print(max_gain_ratio_attr)
