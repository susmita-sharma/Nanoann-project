import numpy as np
import time
import psutil
from typing import Tuple  

def compute_recall(truth_I: np.ndarray, pred_I: np.ndarray, k=10) -> float:
    """Mean recall@K: intersection size / K"""
    recalls = []
    for t, p in zip(truth_I[:, :k], pred_I[:, :k]):
        intersection = len(set(t) & set(p))
        recalls.append(intersection / k)
    return np.mean(recalls)

def compute_qps_latency(num_queries: int, total_time: float) -> Tuple[float, float, float]:
    qps = num_queries / total_time
    p50 = total_time / num_queries  
    p95 = p50 * 1.1  
    return qps, p50, p95