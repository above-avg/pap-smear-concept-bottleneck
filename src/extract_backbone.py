import torch

ckpt = torch.load("models/dino/checkpoint.pth", map_location="cpu")

teacher = ckpt["teacher"]

torch.save(teacher, "models/dino/dino_backbone.pth")

print("Backbone saved.")
