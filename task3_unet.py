import os, glob, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(24142732)  # student ID as seed, reproducible
DATA_DIR = "nuclei_dataset"
OUT = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- Dataset ----------------
class NucleiDataset(Dataset):
    def __init__(self, split, size=128):
        # 128x128 keeps CPU training tractable while remaining a fair comparison;
        # bump back to 256 if training on GPU.
        self.img_paths = sorted(glob.glob(f"{DATA_DIR}/{split}/images/*.png"))
        self.mask_paths = sorted(glob.glob(f"{DATA_DIR}/{split}/masks/*.png"))
        assert len(self.img_paths) == len(self.mask_paths)
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize((self.size, self.size), Image.NEAREST)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, os.path.basename(self.img_paths[idx])

# ---------------- Small U-Net ----------------
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class SmallUNet(nn.Module):
    """Compact 3-level U-Net, sized for a small biomedical dataset trained on CPU."""
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)  # logits

# ---------------- Losses & metrics ----------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(1)
    union = probs.sum(1) + targets.sum(1)
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)

@torch.no_grad()
def dice_iou(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    preds_f = preds.view(preds.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)
    inter = (preds_f * targets_f).sum(1)
    union = preds_f.sum(1) + targets_f.sum(1)
    dice = (2*inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice.mean().item(), iou.mean().item()

# ---------------- Train ----------------
def train(epochs=15, lr=1e-3, batch_size=4, img_size=128):
    train_ds = NucleiDataset("train", size=img_size)
    val_ds = NucleiDataset("val", size=img_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SmallUNet(base=16).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    t0 = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for img, mask, _ in train_dl:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            logits = model(img)
            loss = bce_dice_loss(logits, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, val_dice, val_iou, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for img, mask, _ in val_dl:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                logits = model(img)
                loss = bce_dice_loss(logits, mask)
                val_loss += loss.item() * img.size(0)
                d, i = dice_iou(logits, mask)
                val_dice += d * img.size(0); val_iou += i * img.size(0)
                n += img.size(0)
        val_loss /= n; val_dice /= n; val_iou /= n
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        print(f"Epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | val_iou {val_iou:.4f}")

    print(f"Training took {time.time()-t0:.1f}s on {DEVICE}")
    return model, history, train_ds, val_ds

if __name__ == "__main__":
    model, history, train_ds, val_ds = train(epochs=15, img_size=128)
    torch.save(model.state_dict(), f"{OUT}/unet_state_dict.pt")
    pd.DataFrame(history).to_csv(f"{OUT}/task3_training_history.csv", index=False)

    # loss & dice curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE+Dice loss"); axes[0].legend(); axes[0].set_title("Loss curves")
    axes[1].plot(history["epoch"], history["val_dice"], label="val Dice")
    axes[1].plot(history["epoch"], history["val_iou"], label="val IoU")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend(); axes[1].set_title("Val Dice / IoU")
    plt.tight_layout()
    plt.savefig(f"{OUT}/task3_loss_dice_curves.png", dpi=130)
    plt.close()

    print(f"\nFinal val Dice: {history['val_dice'][-1]:.4f}, Final val IoU: {history['val_iou'][-1]:.4f}")

    # side-by-side panels for 3 validation images
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    with torch.no_grad():
        for row in range(3):
            img, mask, name = val_ds[row]
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()
            axes[row,0].imshow(img.squeeze().numpy(), cmap="gray"); axes[row,0].set_title(f"Input ({name})", fontsize=9); axes[row,0].axis("off")
            axes[row,1].imshow(mask.squeeze().numpy(), cmap="gray"); axes[row,1].set_title("Ground truth", fontsize=9); axes[row,1].axis("off")
            axes[row,2].imshow(pred, cmap="gray"); axes[row,2].set_title("U-Net prediction", fontsize=9); axes[row,2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/task3_val_panels.png", dpi=130)
    plt.close()
    print("Saved outputs/task3_loss_dice_curves.png and outputs/task3_val_panels.png")
