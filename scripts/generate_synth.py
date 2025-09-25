import numpy as np
N, D = 10000, 128
vectors = np.random.randn(N, D).astype(np.float32)
ids = np.arange(N, dtype=np.int64)
np.save('synth_train.npy', vectors)
np.save('synth_ids.npy', ids)
queries = np.random.randn(100, D).astype(np.float32)
np.save('synth_query.npy', queries)