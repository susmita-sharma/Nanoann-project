import numpy as np
from scipy.cluster.vq import kmeans2
from concurrent.futures import ThreadPoolExecutor
from nanoann.distance import get_distance_func
from typing import Tuple

class IVFFlatIndex:
    def __init__(self, metric="l2", nlist=100, nprobe=10, num_threads=4):
        self.metric = metric
        self.dist_func = get_distance_func(metric)
        self.nlist = nlist
        self.nprobe = nprobe
        self.num_threads = num_threads
        self.centroids = None
        self.inv_lists = None  
        self.vectors = None
        self.ids = None
        self.cluster_assign = None  

    def build(self, vectors: np.ndarray, ids: np.ndarray):
        assert vectors.shape[0] == ids.shape[0]
        self.vectors = vectors
        self.ids = ids
        
        print("Training k-means...")
        self.centroids, self.cluster_assign = kmeans2(vectors, k=self.nlist, minit='points')
        
        self.inv_lists = [[] for _ in range(self.nlist)]
        for i, cl in enumerate(self.cluster_assign):
            self.inv_lists[cl].append(i)  
        self.inv_lists = [np.array(lst, dtype=np.int64) for lst in self.inv_lists]

    def search(self, queries: np.ndarray, k=10) -> Tuple[np.ndarray, np.ndarray]:
        if self.centroids is None:
            raise ValueError("Index not built")
        
        def single_search(q):
            cent_dists = self.dist_func(q, self.centroids)
            top_centroids = np.argpartition(cent_dists, self.nprobe)[:self.nprobe]
            sorted_top = top_centroids[np.argsort(cent_dists[top_centroids])]
            
            cand_indices = np.concatenate([self.inv_lists[c] for c in sorted_top])
            if len(cand_indices) == 0:
                return np.full(k, np.inf), np.full(k, -1)
            
            cand_vectors = self.vectors[cand_indices] 
            dists = self.dist_func(q, cand_vectors)
            top_k_idx = np.argpartition(dists, min(k, len(dists)))[:k]
            sorted_k = top_k_idx[np.argsort(dists[top_k_idx])]
            return dists[sorted_k], self.ids[cand_indices[sorted_k]]
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(single_search, queries))
        
        D = np.array([r[0] for r in results])
        I = np.array([r[1] for r in results])
        return D, I