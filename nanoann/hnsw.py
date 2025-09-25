import numpy as np
import hnswlib
from typing import Tuple

class HNSW:
    def __init__(self, metric="l2", M=16, ef_construction=200, ef=50, num_threads=4):
        self.metric = "l2" if metric == "l2" else "cosine"
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.vectors = None
        self.ids = None
        self.index = None

    def build(self, vectors: np.ndarray, ids: np.ndarray):
        assert vectors.shape[0] == ids.shape[0]
        assert vectors.dtype == np.float32
        assert ids.dtype == np.int64
        self.vectors = vectors.copy()
        self.ids = ids.copy()
        self.index = hnswlib.Index(space=self.metric, dim=vectors.shape[1])
        self.index.init_index(max_elements=len(vectors), ef_construction=self.ef_construction, M=self.M)
        self.index.set_num_threads(self.num_threads)
        self.index.add_items(vectors, ids)  
        print(f"✅ Built HNSW index with {len(vectors)} elements")

    def search(self, queries: np.ndarray, k=10, ef=None) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("Index not built")
        ef = ef or self.ef
        self.index.set_ef(ef)
        distances, indices = self.index.knn_query(queries, k=k)
        indices = indices.astype(np.int64, copy=False)
        return distances, indices  
