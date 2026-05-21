import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from src.inference.model import load_local_tokenizer

# File Paths
DATASET_CONFIG = Path("configs/datasets.json")
MODEL_CONFIG = Path("configs/models.json")
OUTPUT_DIR = Path("data/corpus_freqs")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_analysis(dataset_key: str, max_docs: int):
    datasets_spec = load_json(DATASET_CONFIG)
    models_spec = load_json(MODEL_CONFIG)

    spec = datasets_spec[dataset_key]
    model_ids = [m for m in models_spec if not m.startswith("_")]

    # --- Step 1: Read once ---
    print(f"Loading {max_docs} documents from {dataset_key}...")
    dataset = load_dataset(
        path=spec["dataset_name"],
        name=spec.get("dataset_config"),
        split=spec["split"],
        streaming=True
    ).take(max_docs)

    # Store only the text field in memory to minimize footprint
    corpus = []
    for sample in dataset:
        text = sample.get(spec["text_field"], "")
        if text:
            corpus.append(text)

    if not corpus:
        print("No data found.")
        return

    # --- Step 2: Tokenize per model ---
    for model_id in model_ids:
        print(f"Processing Model: {model_id}")
        model_path = Path(models_spec[model_id]["path"])

        try:
            tokenizer = load_local_tokenizer(model_path)
        except Exception as exc:
            print(f"Skipping {model_id}: failed to load tokenizer from {model_path}: {exc}")
            continue

        counts = Counter()
        total_tokens = 0

        for text in corpus:
            ids = tokenizer.encode(text, add_special_tokens=False)
            counts.update(ids)
            total_tokens += len(ids)

        if total_tokens > 0:
            out_path = OUTPUT_DIR / dataset_key / f"{model_id}_{max_docs}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            results = {
                "model": model_id,
                "freq_dataset": dataset_key,
                "total_tokens": total_tokens,
                "frequencies": [
                    {"id": int(tid), "f": count / total_tokens}
                    for tid, count in sorted(counts.items())
                ]
            }

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_analysis(dataset_key="openwebtext", max_docs=5000)
