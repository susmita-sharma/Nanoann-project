import json
import matplotlib.pyplot as plt

# Load results
methods = [
    ('BruteForce', 'results/brute.json'),
    ('IVF nprobe=1', 'results/ivf_nprobe_1.json'),
    ('IVF nprobe=4', 'results/ivf_nprobe_4.json'),
    ('IVF nprobe=16', 'results/ivf_nprobe_16.json'),
    ('IVF nprobe=32', 'results/ivf_nprobe_32.json'),
    ('HNSW ef=10', 'results/hnsw_ef_10.json'),
    ('HNSW ef=50', 'results/hnsw_ef_50.json'),
    ('HNSW ef=100', 'results/hnsw_ef_100.json')
]

recalls = []
qps = []
labels = []

for label, result_file in methods:
    with open(result_file) as f:
        data = json.load(f)
        recalls.append(data['recall@10'])
        qps.append(data['qps'])
        labels.append(label)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(qps, recalls, c='blue')
for i, label in enumerate(labels):
    plt.annotate(label, (qps[i], recalls[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
plt.plot(qps, recalls, 'b-', alpha=0.3)
plt.xlabel('Queries Per Second (QPS)')
plt.ylabel('Recall@10')
plt.title('Recall@10 vs. QPS for BruteForce, IVF, and HNSW')
plt.grid(True)
plt.savefig('results/recall_qps_comparison.png')
plt.close()