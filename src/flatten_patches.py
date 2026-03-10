import os
import shutil
from tqdm import tqdm

INPUT_DIR = "patches"
OUTPUT_DIR = "ssl_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for slide in tqdm(os.listdir(INPUT_DIR)):

    slide_path = os.path.join(INPUT_DIR, slide)

    for img in os.listdir(slide_path):

        src = os.path.join(slide_path, img)

        dst = os.path.join(OUTPUT_DIR, f"{slide}_{img}")

        shutil.copy(src, dst)

        count += 1

print("Total patches:", count)