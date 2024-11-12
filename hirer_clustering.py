import numpy as np

# Load distance matrix from a file
def load_distance_matrix(file_path):
    with open(file_path, 'r') as f:
        matrix = [list(map(float, line.split())) for line in f]
    return np.array(matrix)

# Find the two closest clusters
def find_min_distance(matrix, clusters):
    min_distance = float('inf')
    pair = (-1, -1)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = matrix[clusters[i], :][:, clusters[j]].min()  # Minimum distance between clusters
            if dist < min_distance:
                min_distance = dist
                pair = (i, j)
    return pair, min_distance

# Update the distance matrix to reflect merged clusters
def update_distance_matrix(matrix, clusters, cluster_a, cluster_b):
    new_cluster = clusters[cluster_a] + clusters[cluster_b]
    for i in range(len(clusters)):
        if i != cluster_a and i != cluster_b:
            dist = min(matrix[clusters[cluster_a], :][:, clusters[i]].min(),
                       matrix[clusters[cluster_b], :][:, clusters[i]].min())
            matrix[clusters[cluster_a], :][:, clusters[i]] = dist
            matrix[clusters[i], :][:, clusters[cluster_a]] = dist
    return new_cluster

# Hierarchical clustering
def hierarchical_clustering(matrix):
    n = len(matrix)
    clusters = [[i] for i in range(n)]

    # While more than one cluster remains
    while len(clusters) > 1:
        (a, b), min_dist = find_min_distance(matrix, clusters)
        new_cluster = update_distance_matrix(matrix, clusters, a, b)

        # Update clusters
        clusters[a] = new_cluster
        del clusters[b]

        # Print current clusters and their minimum distance
        print(f"Merged clusters: {new_cluster}, Distance: {min_dist}")

    return clusters[0]

# Main program
file_path = 'matrix.txt'  # Path to your distance matrix file
matrix = load_distance_matrix(file_path)
final_cluster = hierarchical_clustering(matrix)
print(f"Final cluster: {final_cluster}")
