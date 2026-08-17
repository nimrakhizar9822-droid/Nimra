"""
Task 2 (part 2): pass the numbers-only summary (from task2_classical.py) to a
local text LLM and request a one-paragraph description + structured JSON.
The model NEVER sees the image, only the regionprops-derived summary text.
Run this on your machine with Ollama running.
"""
import json
import ollama
from task2_classical import load_gray, otsu_segment, feature_table, table_to_summary

MODEL = "llama3.2"  # text-only model; swap for whatever you have pulled
DATA_DIR = "nuclei_dataset"
OUT = "outputs"

LLM_PROMPT_TEMPLATE = """You are given ONLY numeric measurements extracted by classical image
processing (Otsu thresholding + connected-component analysis) from a fluorescence microscopy
image. You have NOT seen the image itself. Do not invent visual details you were not given.

Measurements:
{summary}

Write:
1. One paragraph (2-4 sentences) describing what this object population likely looks like,
   grounded ONLY in the numbers above.
2. Then a JSON object on its own, matching exactly this schema:
{{
  "n_objects": integer,
  "density_class": string,   // one of "sparse", "normal", "dense", "clustered", "uncertain"
  "shape_regularity": string, // one of "regular", "irregular", "mixed", "uncertain"
  "quality_flag": string      // one of "reliable", "check_thresholding", "uncertain"
}}

Respond with the paragraph first, then a line "---JSON---", then the JSON object only."""


def call_text_llm(summary_text):
    prompt = LLM_PROMPT_TEMPLATE.format(summary=summary_text)
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


if __name__ == "__main__":
    image_id = "test_004"
    gray = load_gray(f"{DATA_DIR}/test/images/{image_id}.png")
    labeled, binary, thresh = otsu_segment(gray)
    df = feature_table(labeled, gray.astype(float) / 255.0)
    summary = table_to_summary(df, image_id)
    print("Numbers-only summary sent to LLM:\n", summary, "\n")

    try:
        out = call_text_llm(summary)
        print("=== LLM response ===")
        print(out)
        with open(f"{OUT}/task2_llm_output_{image_id}.txt", "w") as f:
            f.write(f"SUMMARY SENT TO LLM:\n{summary}\n\nLLM RESPONSE:\n{out}")
        print(f"\nSaved outputs/task2_llm_output_{image_id}.txt")
    except Exception as e:
        print("Could not reach Ollama on this machine. Make sure `ollama serve` is running")
        print("and you have pulled a text model, e.g.: ollama pull llama3.2")
        print(f"Underlying error: {e}")
