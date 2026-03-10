import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dino"))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from vision_transformer import vit_small


device = "cuda" if torch.cuda.is_available() else "cpu"

model = vit_small(patch_size=16)

weights = torch.load(
    "models/dino/dino_backbone.pth",
    map_location="cpu"
)

model.load_state_dict(weights, strict=False)
model.eval().to(device)


transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
])

patch_dir = "ssl_dataset/all"

patch_files = sorted([
    f for f in os.listdir(patch_dir)
    if f.endswith(".png")
])

print("Total patches:", len(patch_files))


embeddings = []

for file in tqdm(patch_files):

    img_path = os.path.join(patch_dir, file)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)

    embeddings.append(feat.cpu().numpy())

embeddings = np.concatenate(embeddings)

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/patch_embeddings.npy", embeddings)

print("Embedding matrix shape:", embeddings.shape)