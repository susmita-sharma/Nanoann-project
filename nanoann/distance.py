import numpy as np

def compute_l2(query: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute L2 distances: query (1,D) vs points (N,D) -> (N,)"""
    return np.sqrt(np.sum((points - query)**2, axis=1))

def compute_cosine(query: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute cosine distances: query (1,D) vs points (N,D) -> (N,)"""
    query_norm = np.linalg.norm(query)
    points_norms = np.linalg.norm(points, axis=1)
    dots = np.dot(points, query.T).squeeze()
    sim = dots / (query_norm * points_norms + 1e-10)  
    return 1 - sim

def get_distance_func(metric: str):
    if metric == "l2":
        return compute_l2
    elif metric == "cosine":
        return compute_cosine
    else:
        raise ValueError("Unsupported metric")
    

import numpy as np
from nanoann.distance import compute_l2, compute_cosine

pts = np.array([[1, 0], [0, 1]], dtype=np.float32)
q = np.array([1, 0], dtype=np.float32)
print(compute_l2(q, pts)) 
print(compute_cosine(q, pts))  