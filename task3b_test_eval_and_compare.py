"""Evaluate trained U-Net on the held-out TEST split, and compare against
Otsu+morphology on the same images/masks -> answers report Q2 & Q3."""
import glob, os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk, remove_small_objects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task3_unet import SmallUNet, dice_iou, DEVICE

OUT = "outputs"
DATA_DIR = "nuclei_dataset"
IMG_SIZE = 128

def load_gray(path, size=IMG_SIZE):
    return np.array(Image.open(path).convert("L").resize((size,size), Image.BILINEAR), dtype=np.float32)/255.0

def load_mask(path, size=IMG_SIZE):
    return (np.array(Image.open(path).convert("L").resize((size,size), Image.NEAREST)) > 127).astype(np.float32)

def otsu_binary(gray):
    t = threshold_otsu(gray)
    b = gray > t
    b = opening(b, disk(1))
    b = closing(b, disk(1))
    b = remove_small_objects(b, min_size=8)
    return b.astype(np.float32)

def dice_np(a, b, eps=1e-6):
    inter = (a*b).sum()
    return (2*inter+eps)/(a.sum()+b.sum()+eps)

def iou_np(a, b, eps=1e-6):
    inter = (a*b).sum()
    union = a.sum()+b.sum()-inter
    return (inter+eps)/(union+eps)

model = SmallUNet(base=16).to(DEVICE)
model.load_state_dict(torch.load(f"{OUT}/unet_state_dict.pt", map_location=DEVICE))
model.eval()

img_paths = sorted(glob.glob(f"{DATA_DIR}/test/images/*.png"))
mask_paths = sorted(glob.glob(f"{DATA_DIR}/test/masks/*.png"))

rows = []
worst_unet, best_otsu_vs_unet = None, None
for ip, mp in zip(img_paths, mask_paths):
    image_id = os.path.splitext(os.path.basename(ip))[0]
    gray = load_gray(ip)
    gt = load_mask(mp)

    with torch.no_grad():
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(t)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()

    otsu_pred = otsu_binary(gray)

    d_unet, i_unet = dice_np(pred, gt), iou_np(pred, gt)
    d_otsu, i_otsu = dice_np(otsu_pred, gt), iou_np(otsu_pred, gt)
    rows.append({"image_id": image_id, "unet_dice": d_unet, "unet_iou": i_unet,
                 "otsu_dice": d_otsu, "otsu_iou": i_otsu})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/task3_test_unet_vs_otsu.csv", index=False)
print(df.round(4))
print("\nMean U-Net  Dice/IoU:", df.unet_dice.mean().round(4), df.unet_iou.mean().round(4))
print("Mean Otsu   Dice/IoU:", df.otsu_dice.mean().round(4), df.otsu_iou.mean().round(4))

# find one image where U-Net beats Otsu by the largest margin, and vice versa
df["unet_minus_otsu"] = df["unet_dice"] - df["otsu_dice"]
best_for_unet = df.loc[df["unet_minus_otsu"].idxmax()]
best_for_otsu = df.loc[df["unet_minus_otsu"].idxmin()]
print("\nImage where U-Net wins most:", best_for_unet.image_id, "diff", round(best_for_unet.unet_minus_otsu,4))
print("Image where Otsu wins most (or U-Net worst relative):", best_for_otsu.image_id, "diff", round(best_for_otsu.unet_minus_otsu,4))

# visualize the two extreme examples: gt vs unet vs otsu
fig, axes = plt.subplots(2, 3, figsize=(9,6))
for row, image_id in enumerate([best_for_unet.image_id, best_for_otsu.image_id]):
    ip = f"{DATA_DIR}/test/images/{image_id}.png"
    mp = f"{DATA_DIR}/test/masks/{image_id}.png"
    gray = load_gray(ip); gt = load_mask(mp)
    with torch.no_grad():
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pred = (torch.sigmoid(model(t)) > 0.5).float().cpu().squeeze().numpy()
    otsu_pred = otsu_binary(gray)
    axes[row,0].imshow(gt, cmap="gray"); axes[row,0].set_title(f"{image_id} GT"); axes[row,0].axis("off")
    axes[row,1].imshow(pred, cmap="gray"); axes[row,1].set_title("U-Net"); axes[row,1].axis("off")
    axes[row,2].imshow(otsu_pred, cmap="gray"); axes[row,2].set_title("Otsu"); axes[row,2].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/task3_unet_vs_otsu_examples.png", dpi=130)
plt.close()
print("Saved outputs/task3_unet_vs_otsu_examples.png")
