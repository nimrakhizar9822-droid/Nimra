"""
Task 1 (part 2): send a representative image to a local multimodal model
(llama3.2-vision) via Ollama, comparing a NAIVE prompt against an ENGINEERED
structured prompt. Run this file directly on your machine, where Ollama is
already installed and running (`ollama serve`), after pulling the model:

    ollama pull llama3.2-vision

If Ollama is not reachable, this script prints a clear message instead of
crashing, so the rest of the notebook can still run end-to-end.
"""
import os, json, glob
import ollama

MODEL = "llama3.2-vision"
DATA_DIR = "nuclei_dataset"
OUT = "outputs"

REPRESENTATIVE_IMAGE = sorted(glob.glob(f"{DATA_DIR}/train/images/*.png"))[4]  # train_004: dense regime

# --- Naive prompt: no anchoring, no output format, invites diagnostic overreach ---
NAIVE_PROMPT = "What do you see in this microscopy image?"

# --- Engineered structured prompt ---
# Design choices (discuss these in the report):
#  1. Explicitly anchors the model as a DESCRIPTIVE assistant, not a diagnostic one,
#     to reduce the chance it invents a clinical diagnosis it has no basis for.
#  2. Forces a strict JSON schema so downstream code can parse the output reliably
#     instead of scraping free text (this is what "auditable" means in this assignment).
#  3. Explicitly permits "uncertain" as a valid value for any field, which gives the
#     model a documented way to abstain instead of guessing/hallucinating.
#  4. Asks it to ground notable_features only in what is visually present.
STRUCTURED_PROMPT = """You are an image-description assistant supporting a research pipeline.
You are NOT a diagnostic tool and must not provide clinical diagnoses, disease names,
or treatment suggestions. Only describe what is visually present in the image.

Describe this fluorescence microscopy image and respond with ONLY a single JSON object,
no other text, matching exactly this schema:

{
  "modality": string,          // e.g. "fluorescence microscopy" or "uncertain"
  "tissue_type": string,       // best guess at tissue/cell type, or "uncertain"
  "notable_features": string,  // 1-2 sentences on shape, distribution, density of visible objects
  "image_quality": string      // one of: "good", "noisy", "low_contrast", "uncertain"
}

If you are not confident about a field, use the string "uncertain" for that field rather
than guessing. Do not include any text before or after the JSON object."""


def call_vlm(image_path, prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
    )
    return response["message"]["content"]


def try_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # model sometimes wraps JSON in ```json fences despite instructions
        cleaned = text.strip().strip("`")
        cleaned = cleaned.replace("json\n", "", 1) if cleaned.startswith("json\n") else cleaned
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"Representative image: {REPRESENTATIVE_IMAGE}\n")

    try:
        print("=== NAIVE PROMPT ===")
        naive_out = call_vlm(REPRESENTATIVE_IMAGE, NAIVE_PROMPT)
        print(naive_out)

        print("\n=== STRUCTURED PROMPT (run 1) ===")
        structured_out_1 = call_vlm(REPRESENTATIVE_IMAGE, STRUCTURED_PROMPT)
        print(structured_out_1)

        print("\n=== STRUCTURED PROMPT (run 2, same image+prompt, to show non-determinism) ===")
        structured_out_2 = call_vlm(REPRESENTATIVE_IMAGE, STRUCTURED_PROMPT)
        print(structured_out_2)

        parsed_1 = try_parse_json(structured_out_1)
        parsed_2 = try_parse_json(structured_out_2)

        results = {
            "image": REPRESENTATIVE_IMAGE,
            "naive_prompt": NAIVE_PROMPT,
            "naive_output": naive_out,
            "structured_prompt": STRUCTURED_PROMPT,
            "structured_output_run1": structured_out_1,
            "structured_output_run1_valid_json": parsed_1 is not None,
            "structured_output_run2": structured_out_2,
            "structured_output_run2_valid_json": parsed_2 is not None,
            "runs_identical": structured_out_1.strip() == structured_out_2.strip(),
        }
        with open(f"{OUT}/task1_vlm_outputs.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRuns identical: {results['runs_identical']}")
        print("Saved outputs/task1_vlm_outputs.json")

    except Exception as e:
        print("Could not reach Ollama / llama3.2-vision on this machine.")
        print("Make sure `ollama serve` is running and you have run:")
        print("    ollama pull llama3.2-vision")
        print(f"Underlying error: {e}")
