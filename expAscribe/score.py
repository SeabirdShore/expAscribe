from sklearn.metrics import precision_score, recall_score, f1_score
import networkx as nx
import numpy as np

def score_final_adj_matrix(final_adj_matrix:np.ndarray, y_meta:np.ndarray):
    
    y_true_flat = y_meta.flatten()
    y_pred_flat = final_adj_matrix.flatten()

    y_true_binary = (y_true_flat > 0).astype(int)
    y_pred_binary = (y_pred_flat > 0).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return precision, recall, f1


def ratio_of_paths(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate the ratio of paths in B that exist in A.

    Parameters:
    - A: A numpy array representing the adjacency matrix of a graph.
    - B: A numpy array representing the adjacency matrix of a graph.

    Returns:
    - y/x: The ratio value.
    """

    def has_path(graph: nx.DiGraph, start: int, end: int) -> bool:
        """Check if there is a directed path from start node to end node in the graph."""
        try:
            # Check if there is a path
            return nx.has_path(graph, start, end)
        except nx.NetworkXError:
            # If the graph is not connected, return False
            return False

    # Create a directed graph from A
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    # Calculate the number of 1s in B
    x = np.sum(B)

    # Initialize y to 0
    y = 0

    # Iterate through each element in B
    rows, cols = np.where(B == 1)
    for i, j in zip(rows, cols):
        if has_path(G, i, j):
            y += 1

    # Return y/x
    return y / x if x != 0 else 0.0
