import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "nuclei_dataset"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

def load_gray_resized(path, size=(256,256)):
    im = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.array(im)

train_imgs = sorted(glob.glob(f"{DATA_DIR}/train/images/*.png"))
print("train images:", len(train_imgs))

# sample grid of 6 images (grayscale, resized)
fig, axes = plt.subplots(2, 3, figsize=(9,6))
sample_paths = train_imgs[:6]
for ax, p in zip(axes.ravel(), sample_paths):
    g = load_gray_resized(p)
    ax.imshow(g, cmap="gray")
    ax.set_title(os.path.basename(p), fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/task1_sample_grid.png", dpi=130)
plt.close()

# intensity histogram over a batch of grayscale images
all_pixels = []
for p in train_imgs[:20]:
    g = load_gray_resized(p)
    all_pixels.append(g.ravel())
all_pixels = np.concatenate(all_pixels)

plt.figure(figsize=(6,4))
plt.hist(all_pixels, bins=50, color="slateblue")
plt.title("Grayscale intensity histogram (20 train images, 256x256)")
plt.xlabel("Pixel intensity (0-255)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT}/task1_intensity_hist.png", dpi=130)
plt.close()

print("mean:", all_pixels.mean(), "std:", all_pixels.std())
print("Saved sample grid + histogram to outputs/")
