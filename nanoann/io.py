import json
import os
import hashlib
import numpy as np
import nanoann
import hnswlib

def save_index(index, path: str, mmap=False):  
    os.makedirs(path, exist_ok=True)
    params = {
        'algo': type(index).__name__,
        'metric': index.metric,
        'N': len(index.ids),
        'D': index.vectors.shape[1],
    }
    if hasattr(index, 'nlist'):
        params.update({'nlist': index.nlist, 'nprobe': index.nprobe})
    if hasattr(index, 'M'):  
        params.update({'M': index.M, 'ef_construction': index.ef_construction, 'ef': index.ef})

    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(params, f)

    np.save(os.path.join(path, 'vectors.npy'), index.vectors)
    np.save(os.path.join(path, 'ids.npy'), index.ids)

    if hasattr(index, 'centroids'):  
        np.save(os.path.join(path, 'centroids.npy'), index.centroids)
        np.save(os.path.join(path, 'cluster_assign.npy'), index.cluster_assign)
        for i, lst in enumerate(index.inv_lists):
            np.save(os.path.join(path, f'inv_list_{i}.npy'), lst)

    if params['algo'] == 'HNSW':
        index.index.save_index(os.path.join(path, "hnsw.bin"))

    checksum = hashlib.sha256()
    for fname in sorted(os.listdir(path)):
        with open(os.path.join(path, fname), 'rb') as f:
            checksum.update(f.read())
    with open(os.path.join(path, 'checksum.sha256'), 'w') as f:
        f.write(checksum.hexdigest())

def load_index(path: str, mmap=False) -> 'nanoann.Index':
    with open(os.path.join(path, 'params.json'), 'r') as f:
        params = json.load(f)

    print("Skipping checksum verification for debugging")

    memmap_mode = 'r' if mmap else None
    vectors = np.load(os.path.join(path, 'vectors.npy'), mmap_mode=memmap_mode)
    ids = np.load(os.path.join(path, 'ids.npy'), mmap_mode=memmap_mode)

    if params['algo'] == 'BruteForceIndex':
        idx = nanoann.Index(algo="brute", metric=params['metric'])
        idx.vectors = vectors
        idx.ids = ids
        idx.build(vectors, ids)  
    elif params['algo'] == 'IVFFlatIndex':
        idx = nanoann.Index(algo="ivf", metric=params['metric'], nlist=params['nlist'], nprobe=params['nprobe'])
        idx.vectors = vectors  
        idx.ids = ids         
        idx.centroids = np.load(os.path.join(path, 'centroids.npy'), mmap_mode=memmap_mode)
        idx.cluster_assign = np.load(os.path.join(path, 'cluster_assign.npy'), mmap_mode=memmap_mode)
        idx.inv_lists = []
        for i in range(params['nlist']):
            lst = np.load(os.path.join(path, f'inv_list_{i}.npy'), mmap_mode=memmap_mode)
            idx.inv_lists.append(lst)
        idx.build(vectors, ids)  
    elif params['algo'] == 'HNSW':
        idx = nanoann.Index(algo="hnsw", metric=params['metric'], M=params.get('M'),
                            ef_construction=params.get('ef_construction'), ef=params.get('ef'))
        idx.vectors = vectors
        idx.ids = ids
        if idx.index is None:
            idx.index = hnswlib.Index(space=idx.metric, dim=vectors.shape[1])
        idx.index.load_index(os.path.join(path, "hnsw.bin"))
        idx.index.set_ef(idx.ef)
    else:
        raise ValueError("Unknown algo")

    return idx



