from collections import Counter
import pandas as pd

def calculate_gini_index(data, attribute_name, target_name):
    """Calculate the Gini index for a specific attribute in the dataset."""
    total_samples = len(data)
    attribute_values = {}

    # Group the data by attribute values
    for _, row in data.iterrows():
        attribute_value = row[attribute_name]
        target_value = row[target_name]

        if attribute_value not in attribute_values:
            attribute_values[attribute_value] = []
        attribute_values[attribute_value].append(target_value)

    # Calculate Gini index
    gini_index = 0.0
    for values in attribute_values.values():
        subset_size = len(values)
        if subset_size == 0:
            continue
        score = 0.0
        class_counts = Counter(values)

        # Calculate score for each class in the subset
        for count in class_counts.values():
            proportion = count / subset_size
            score += proportion * proportion

        # Weighted Gini index for the attribute
        gini_index += (1 - score) * (subset_size / total_samples)

    return gini_index

def find_minimum_gini(data, target_name, exclude_columns=None):
    """
    Find the attribute with the minimum Gini index for any dataset.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        target_name (str): The name of the target column.
        exclude_columns (list, optional): List of columns to exclude from Gini calculation.
    """
    if exclude_columns is None:
        exclude_columns = []
    exclude_columns.append(target_name)

    # Filter attribute names excluding target and specified columns
    attribute_names = data.columns.drop(exclude_columns)
    gini_indices = {}

    # Calculate Gini index for each attribute
    for attribute in attribute_names:
        gini_index = calculate_gini_index(data, attribute, target_name)
        gini_indices[attribute] = gini_index

    # Find the attribute with the minimum Gini index
    min_gini_attribute = min(gini_indices, key=gini_indices.get)

    # Display results in table format
    print(f"{'Attribute':<15}{'Gini Index':<10}")
    print("-" * 25)
    for attribute, gini in gini_indices.items():
        print(f"{attribute:<15}{gini:<10.4f}")

    print("\nAttribute with minimum Gini index:")
    print(f"{min_gini_attribute}: {gini_indices[min_gini_attribute]:.4f}")

# Example usage:
# Assuming `df` is your dataset and 'PlayTennis' is the target column
find_minimum_gini(df, target_name='PlayTennis', exclude_columns=['Day'])
