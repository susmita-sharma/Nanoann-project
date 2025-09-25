import argparse
import json
import time
import numpy as np
import psutil
from nanoann import Index, save_index, load_index
from nanoann.metrics import compute_recall, compute_qps_latency
import os

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Build
    build_p = subparsers.add_parser('build')
    build_p.add_argument('--algo', required=True)
    build_p.add_argument('--metric', required=True)
    build_p.add_argument('--data', required=True)
    build_p.add_argument('--ids', required=True)
    build_p.add_argument('--out', required=True)
    build_p.add_argument('--nlist', type=int)
    build_p.add_argument('--nprobe', type=int)
    build_p.add_argument('--M', type=int) 
    build_p.add_argument('--ef_construction', type=int)  
    build_p.add_argument('--ef', type=int)  
    
    # Search
    search_p = subparsers.add_parser('search')
    search_p.add_argument('--index', required=True)
    search_p.add_argument('--queries', required=True)
    search_p.add_argument('--k', type=int, default=10)
    search_p.add_argument('--out', required=True)
    
    # Eval
    eval_p = subparsers.add_parser('eval')
    eval_p.add_argument('--truth', required=True)
    eval_p.add_argument('--run', required=True)
    eval_p.add_argument('--index', required=True)
    eval_p.add_argument('--report', required=True)
    
    # Info
    info_p = subparsers.add_parser('info')
    info_p.add_argument('--index', required=True)
    
    # Check
    check_p = subparsers.add_parser('check')
    check_p.add_argument('--index', required=True)
    
    args = parser.parse_args()
    
    if args.command == 'build':
        vectors = np.load(args.data)
        ids = np.load(args.ids)
        params = {}
        if args.nlist: params['nlist'] = args.nlist
        if args.nprobe: params['nprobe'] = args.nprobe
        if args.M: params['M'] = args.M
        if args.ef_construction: params['ef_construction'] = args.ef_construction
        if args.ef: params['ef'] = args.ef
        idx = Index(algo=args.algo, metric=args.metric, **params)
        start = time.time()
        idx.build(vectors, ids)
        build_time = time.time() - start

        idx.vectors = vectors 
        idx.ids = ids         
        save_index(idx, args.out)
        print(f"Built in {build_time:.2f}s")
    
    elif args.command == 'search':
        idx = load_index(args.index)
        queries = np.load(args.queries)
        start_ram = psutil.Process().memory_info().rss / 1024**2
        start = time.time()
        D, I = idx.search(queries, k=args.k)
        latency = time.time() - start
        end_ram = psutil.Process().memory_info().rss / 1024**2
        with open(args.out, 'w') as f:
            json.dump({'D': D.tolist(), 'I': I.tolist(), 'latency': latency, 'ram_delta': end_ram - start_ram}, f)
    
    elif args.command == 'eval':
        with open(args.truth, 'r') as f:
            truth = json.load(f)
        with open(args.run, 'r') as f:
            run = json.load(f)
        recall1 = compute_recall(np.array(truth['I']), np.array(run['I']), k=1)
        recall10 = compute_recall(np.array(truth['I']), np.array(run['I']), k=10)
        qps, p50, p95 = compute_qps_latency(len(run['I']), run['latency'])
        index_size = sum(os.path.getsize(os.path.join(args.index, f)) for f in os.listdir(args.index)) / 1024**2
        report = {
            'recall@1': recall1,
            'recall@10': recall10,
            'qps': qps,
            'p50_latency': p50,
            'p95_latency': p95,
            'index_size_mb': index_size,
            'ram_mb': run.get('ram_delta', 0)
        }
        with open(args.report, 'w') as f:
            json.dump(report, f)
    
    elif args.command == 'info':
        with open(os.path.join(args.index, 'params.json'), 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    
    elif args.command == 'check':
        try:
            load_index(args.index)
            print("OK")
        except ValueError as e:
            print(f"Failed: {e}")

if __name__ == '__main__':
    main()


