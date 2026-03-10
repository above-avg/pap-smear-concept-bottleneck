import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "data/cervix93/edf"
OUTPUT_DIR = "patches"

PATCH_SIZE = 224
STRIDE = 112

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_background(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    tissue_pixels = np.sum(gray < 220)
    tissue_ratio = tissue_pixels / gray.size

    return tissue_ratio < 0.05

slides = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]

for slide in tqdm(slides):

    slide_path = os.path.join(INPUT_DIR, slide)

    image = cv2.imread(slide_path)

    h, w, _ = image.shape

    slide_name = slide.split(".")[0]
    slide_output = os.path.join(OUTPUT_DIR, slide_name)

    os.makedirs(slide_output, exist_ok=True)

    patch_id = 0

    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):

            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if is_background(patch):
                continue

            patch_path = os.path.join(slide_output, f"{patch_id}.png")

            cv2.imwrite(patch_path, patch)

            patch_id += 1