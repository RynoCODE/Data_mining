import numpy as np

# Function to calculate the Euclidean distance between points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to assign each point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
    return clusters

# Function to update centroids as the mean of points in each cluster
def update_centroids(clusters):
    return np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else np.zeros_like(clusters[0][0]) for cluster in clusters])

# Function to perform K-means clustering
def k_means(data, initial_centroids, max_iters=100, tol=1e-4):
    centroids = np.array(initial_centroids)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return centroids, clusters

# Main script to run K-means on provided 1D, 2D, and 3D data
if __name__ == "__main__":
    # Replace the following lists with your actual data and initial centroids

    # Example 1D data and initial centroids
    data_1d = np.array([[1.0], [2.5], [5.5], [6.0], [8.5], [9.0]])
    initial_centroids_1d = np.array([[2.0], [6.0], [9.0]])

    # Example 2D data and initial centroids
    data_2d = np.array([[1.0, 2.0], [3.0, 3.5], [5.0, 8.0], [7.0, 6.0], [9.0, 9.0], [10.0, 5.0]])
    initial_centroids_2d = np.array([[1.0, 2.0], [5.0, 8.0], [9.0, 9.0]])

    # Example 3D data and initial centroids
    data_3d = np.array([[1.0, 2.0, 1.5], [3.0, 3.5, 2.5], [5.0, 8.0, 6.5], [7.0, 6.0, 5.0], [9.0, 9.0, 8.0], [10.0, 5.0, 3.0]])
    initial_centroids_3d = np.array([[1.0, 2.0, 1.5], [5.0, 8.0, 6.5], [9.0, 9.0, 8.0]])

    # Run K-means on 1D data
    centroids_1d, clusters_1d = k_means(data_1d, initial_centroids_1d)
    print("1D Clusters:")
    for i, cluster in enumerate(clusters_1d):
        print(f"Cluster {i + 1}: {cluster}")
    print("1D Centroids:", centroids_1d)

    # Run K-means on 2D data
    centroids_2d, clusters_2d = k_means(data_2d, initial_centroids_2d)
    print("\n2D Clusters:")
    for i, cluster in enumerate(clusters_2d):
        print(f"Cluster {i + 1}: {cluster}")
    print("2D Centroids:", centroids_2d)

    # Run K-means on 3D data
    centroids_3d, clusters_3d = k_means(data_3d, initial_centroids_3d)
    print("\n3D Clusters:")
    for i, cluster in enumerate(clusters_3d):
        print(f"Cluster {i + 1}: {cluster}")
    print("3D Centroids:", centroids_3d)
