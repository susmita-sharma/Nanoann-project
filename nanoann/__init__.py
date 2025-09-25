from .brute import BruteForceIndex
from .ivf import IVFFlatIndex
from .hnsw import HNSW
from .io import save_index, load_index

def Index(algo="brute", metric="l2", **params):
    if algo == "brute":
        return BruteForceIndex(metric=metric, **params)
    elif algo == "ivf":
        return IVFFlatIndex(metric=metric, **params)
    elif algo == "hnsw":
        return HNSW(metric=metric, **params)  
    else:
        raise ValueError("Unsupported algo")

