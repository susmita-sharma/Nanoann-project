for nprobe in 1 4 16 32; do
    python -m nanoann.cli build --algo ivf --metric l2 --data train.npy --ids train_ids.npy --out idx_ivf_$nprobe --nlist 256 --nprobe $nprobe
    python -m nanoann.cli search --index idx_ivf_$nprobe --queries query.npy --k 10 --out runs/ivf_$nprobe.json
    python -m nanoann.cli eval --truth runs/brute.json --run runs/ivf_$nprobe.json --report results/ivf_$nprobe.json
done