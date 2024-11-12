import pandas as pd
import numpy as np

# Function to calculate the mean of each feature for each class
def calculate_mean(X):
    return np.mean(X, axis=0)

# Function to calculate the variance of each feature for each class
def calculate_variance(X):
    return np.var(X, axis=0)

# Function to calculate the prior probability of each class
def calculate_prior(y):
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / len(y)
    return dict(zip(classes, priors))

# Function to calculate Gaussian probability density function
def gaussian_pdf(x, mean, var):
    eps = 1e-6  # Small constant to avoid division by zero
    coefficient = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponential = np.exp(-((x - mean) ** 2) / (2 * var + eps))
    return coefficient * exponential

# Function to fit the model to the training data
def fit(X, y):
    model = {}
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        model[cls] = {
            "mean": calculate_mean(X_cls),
            "variance": calculate_variance(X_cls),
            "prior": calculate_prior(y)[cls]
        }
    return model

# Function to calculate the posterior probability for each class
def calculate_posterior(X, model):
    posteriors = []
    for cls, params in model.items():
        prior = np.log(params["prior"])  # Take log of prior
        likelihood = np.sum(np.log(gaussian_pdf(X, params["mean"], params["variance"])))
        posterior = prior + likelihood
        posteriors.append((cls, posterior))
    return max(posteriors, key=lambda x: x[1])[0]  # Return class with highest posterior

# Function to predict the class for each sample in X
def predict(X, model):
    predictions = [calculate_posterior(x, model) for x in X]
    return np.array(predictions)

# Example usage:
# Load your data, assume target is last column
data = pd.read_csv("play_tennis.csv")

# Encode categorical variables
for column in data.columns:
    if data[column].dtype == 'object':  # If the column is categorical (strings)
        data[column] = data[column].astype('category').cat.codes

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Train the model
model = fit(X, y)

# Predict on new data (for example, the training data itself)
predictions = predict(X, model)

# Accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
