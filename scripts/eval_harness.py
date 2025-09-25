import json
import glob
import pandas as pd

# Load all result files
result_files = glob.glob('results/*.json')
data = []

for file in result_files:
    with open(file, 'r') as f:
        metrics = json.load(f)
    method = file.split('/')[-1].replace('.json', '')
    data.append({
        'Method': method,
        'Recall@10': metrics['recall@10'],
        'QPS': metrics['qps'],
        'p50 Latency (ms)': metrics['p50_latency'] * 1000,  # Convert s to ms
        'p95 Latency (ms)': metrics['p95_latency'] * 1000,  # Convert s to ms
        'Index Size (MB)': metrics['index_size_mb'],
        'RAM Usage (MB)': metrics['ram_mb']
    })

# Create DataFrame and sort by Recall@10 descending
df = pd.DataFrame(data).sort_values(by='Recall@10', ascending=False)

# Print as Markdown table
print(df.to_markdown(index=False))

# Optional: Save to CSV
df.to_csv('results/full_metrics.csv', index=False)