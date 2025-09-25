import numpy as np
from concurrent.futures import ThreadPoolExecutor
from nanoann.distance import get_distance_func
from typing import Tuple 

class BruteForceIndex:
    def __init__(self, metric="l2", num_threads=4):
        self.metric = metric
        self.dist_func = get_distance_func(metric)
        self.vectors = None
        self.ids = None
        self.num_threads = num_threads

    def build(self, vectors: np.ndarray, ids: np.ndarray):
        assert vectors.shape[0] == ids.shape[0]
        assert vectors.dtype == np.float32
        assert ids.dtype == np.int64
        self.vectors = vectors
        self.ids = ids

    def search(self, queries: np.ndarray, k=10) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise ValueError("Index not built")
        
        def single_search(q):
            dists = self.dist_func(q, self.vectors)
            top_indices = np.argpartition(dists, k)[:k]
            sorted_top = top_indices[np.argsort(dists[top_indices])]
            return dists[sorted_top], self.ids[sorted_top]
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(single_search, queries))
        
        D = np.array([r[0] for r in results])  
        I = np.array([r[1] for r in results])  
        return D, I