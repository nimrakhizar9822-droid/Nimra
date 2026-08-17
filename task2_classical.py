import os, glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from skimage.measure import label, regionprops_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "nuclei_dataset"
OUT = "outputs"

def load_gray(path, size=(256,256)):
    im = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.array(im)

def otsu_segment(gray):
    """Otsu threshold + morphological cleanup -> labeled component image."""
    gray_f = gray.astype(float) / 255.0
    thresh = threshold_otsu(gray_f)
    binary = gray_f > thresh
    # morphological cleanup: opening removes speckle noise, closing fills small gaps
    binary = binary_opening(binary, disk(1))
    binary = binary_closing(binary, disk(1))
    binary = remove_small_objects(binary, min_size=8)
    labeled = label(binary)
    return labeled, binary, thresh

def feature_table(labeled, intensity_img):
    props = regionprops_table(
        labeled, intensity_image=intensity_img,
        properties=("label", "area", "eccentricity", "solidity",
                    "mean_intensity", "perimeter", "major_axis_length", "minor_axis_length")
    )
    return pd.DataFrame(props)

def table_to_summary(df, image_id):
    """Numbers-only natural language summary (no image is passed to the LLM)."""
    if len(df) == 0:
        return f"Image {image_id}: Otsu segmentation found 0 objects after cleanup."
    n = len(df)
    mean_area = df["area"].mean()
    med_area = df["area"].median()
    mean_ecc = df["eccentricity"].mean()
    mean_sol = df["solidity"].mean()
    mean_int = df["mean_intensity"].mean()
    summary = (
        f"Image {image_id}: Otsu thresholding + morphological cleanup detected {n} connected "
        f"components. Mean object area is {mean_area:.1f} px^2 (median {med_area:.1f} px^2). "
        f"Mean eccentricity is {mean_ecc:.2f} (0=circle, 1=line), mean solidity is {mean_sol:.2f} "
        f"(1.0=fully convex, no dents). Mean intensity inside objects is {mean_int:.2f} (0-1 scale)."
    )
    return summary

if __name__ == "__main__":
    test_imgs = sorted(glob.glob(f"{DATA_DIR}/test/images/*.png"))
    rows = []
    example_summary = None
    for p in test_imgs:
        image_id = os.path.splitext(os.path.basename(p))[0]
        gray = load_gray(p)
        labeled, binary, thresh = otsu_segment(gray)
        df = feature_table(labeled, gray.astype(float)/255.0)
        summary = table_to_summary(df, image_id)
        if image_id == "test_004":
            example_summary = summary
            example_df = df.copy()
        rows.append({"image_id": image_id, "n_objects_otsu": len(df),
                     "mean_area": df["area"].mean() if len(df) else 0,
                     "otsu_threshold": thresh})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{OUT}/task2_otsu_summary_per_image.csv", index=False)
    print(summary_df)
    print("\nExample numbers-only summary (test_004), this text is what gets sent to the LLM:\n")
    print(example_summary)
    example_df.to_csv(f"{OUT}/task2_example_feature_table_test_004.csv", index=False)

    # visual check: original / binary mask / labeled overlay for one image
    p = f"{DATA_DIR}/test/images/test_004.png"
    gray = load_gray(p)
    labeled, binary, thresh = otsu_segment(gray)
    fig, axes = plt.subplots(1, 3, figsize=(10,4))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Grayscale input"); axes[0].axis("off")
    axes[1].imshow(binary, cmap="gray"); axes[1].set_title(f"Otsu binary (t={thresh:.3f})"); axes[1].axis("off")
    axes[2].imshow(labeled, cmap="nipy_spectral"); axes[2].set_title(f"Labeled ({labeled.max()} objects)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/task2_otsu_pipeline_test_004.png", dpi=130)
    plt.close()
    print("\nSaved outputs/task2_otsu_pipeline_test_004.png")
