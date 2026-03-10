import numpy as np
import os
from collections import defaultdict


labels = np.load("clusters/cluster_labels.npy")

patch_dir = "ssl_dataset/all"

patch_files = sorted([
    f for f in os.listdir(patch_dir)
    if f.endswith(".png")
])


K = labels.max() + 1

# slide → cluster counts
slide_clusters = defaultdict(lambda: np.zeros(K))

for patch, cluster in zip(patch_files, labels):

    slide_id = patch.split("_")[0]  # frame014

    slide_clusters[slide_id][cluster] += 1


slide_features = {}
for slide in slide_clusters:
    vec = slide_clusters[slide]
    slide_features[slide] = vec / vec.sum()

slides = sorted(slide_features.keys())

X = np.array([slide_features[s] for s in slides])

print("Slides:", len(slides))
print("Feature shape:", X.shape)


os.makedirs("features", exist_ok=True)

np.save("features/slide_features.npy", X)

with open("features/slide_ids.txt", "w") as f:
    for s in slides:
        f.write(s + "\n")

print("Slide feature vectors saved.")