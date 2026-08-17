"""
Task 4: full hybrid pipeline on the unseen TEST images.
  U-Net mask -> regionprops feature table -> LLM structured JSON record -> narrative
Aggregates all records into outputs/task4_hybrid_records.csv

Non-LLM steps (U-Net inference, regionprops) run everywhere.
The LLM step is guarded: if Ollama isn't reachable, quality_flag records that,
and n_objects/mean_area/density_class are still filled in from the real features
so the CSV stays usable even before you run this with Ollama on.
"""
import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image
import torch

from task3_unet import SmallUNet, DEVICE
from skimage.measure import label, regionprops_table

try:
    import ollama
    OLLAMA_MODEL = "llama3.2"
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

DATA_DIR = "nuclei_dataset"
OUT = "outputs"
IMG_SIZE = 128


def load_gray(path, size=IMG_SIZE):
    return np.array(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def unet_mask(model, gray):
    with torch.no_grad():
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pred = (torch.sigmoid(model(t)) > 0.5).float().cpu().squeeze().numpy()
    return pred


def density_class_from_count(n, area_fraction):
    # simple, transparent rule mirroring the dataset's own regime definitions;
    # this is the "source of truth" the LLM's density_class should agree with
    if n < 15:
        return "sparse"
    elif n < 30:
        return "normal"
    elif area_fraction > 0.12 and n >= 30:
        return "dense"
    else:
        return "normal"


def build_llm_record(image_id, n_objects, mean_area, density_class):
    prompt = f"""You are given verified measurements from an automated image-analysis pipeline
(U-Net segmentation -> connected components -> regionprops). Do not contradict these numbers;
your job is only to phrase them and flag quality concerns.

image_id: {image_id}
n_objects: {n_objects}
mean_area_px2: {mean_area:.1f}
density_class (already computed, treat as ground truth): {density_class}

Return ONLY a JSON object:
{{
  "image_id": "{image_id}",
  "n_objects": {n_objects},
  "mean_area": {mean_area:.1f},
  "density_class": "{density_class}",
  "quality_flag": string   // "reliable", "check_segmentation", or "uncertain"
}}
Then on a new line write one short narrative sentence describing this image for a lab notebook."""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


if __name__ == "__main__":
    model = SmallUNet(base=16).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT}/unet_state_dict.pt", map_location=DEVICE))
    model.eval()

    img_paths = sorted(glob.glob(f"{DATA_DIR}/test/images/*.png"))
    records = []

    for ip in img_paths:
        image_id = os.path.splitext(os.path.basename(ip))[0]
        gray = load_gray(ip)
        mask = unet_mask(model, gray)

        labeled = label(mask.astype(int))
        props = regionprops_table(labeled, properties=("label", "area"))
        n_objects = len(props["label"]) if len(props["label"]) else 0
        mean_area = float(np.mean(props["area"])) if n_objects else 0.0
        area_fraction = mask.sum() / mask.size
        dclass = density_class_from_count(n_objects, area_fraction)

        record = {
            "image_id": image_id, "n_objects": n_objects, "mean_area": round(mean_area, 1),
            "density_class": dclass, "quality_flag": "uncertain (LLM not run)",
            "narrative": "",
        }

        if OLLAMA_AVAILABLE:
            try:
                llm_text = build_llm_record(image_id, n_objects, mean_area, dclass)
                record["llm_raw_output"] = llm_text
                # best-effort parse: keep the raw text either way for auditability
                if "quality_flag" in llm_text:
                    for flag in ("reliable", "check_segmentation", "uncertain"):
                        if f'"{flag}"' in llm_text:
                            record["quality_flag"] = flag
                            break
            except Exception as e:
                record["llm_raw_output"] = f"Ollama call failed: {e}"

        records.append(record)
        print(record)

    df = pd.DataFrame(records)
    df.to_csv(f"{OUT}/task4_hybrid_records.csv", index=False)
    print(f"\nSaved {len(df)} records to outputs/task4_hybrid_records.csv")
    if not OLLAMA_AVAILABLE:
        print("\n(ollama package not importable in this environment)")
