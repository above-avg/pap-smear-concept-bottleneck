import numpy as np
import os
import random
import shutil

labels = np.load("clusters/cluster_labels.npy")

patch_dir = "ssl_dataset/all"

patch_files = sorted([
    f for f in os.listdir(patch_dir)
    if f.endswith(".png")
])

K = labels.max() + 1

print("Clusters:", K)

# create output directory
os.makedirs("clusters/visualization", exist_ok=True)

for k in range(K):

    cluster_dir = f"clusters/visualization/cluster_{k}"
    os.makedirs(cluster_dir, exist_ok=True)

    indices = np.where(labels == k)[0]

    
    selected = random.sample(list(indices), min(20, len(indices)))

    for idx in selected:

        src = os.path.join(patch_dir, patch_files[idx])
        dst = os.path.join(cluster_dir, patch_files[idx])

        shutil.copy(src, dst)

print("Cluster visualization images saved.")