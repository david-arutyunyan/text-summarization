import os
import json
from datasets import load_dataset


def extract_legal_eng_examples(output_path, max_len=400, max_examples=30):
    dataset = load_dataset("jonathanli/eurlex", split="train")
    selected = []

    for idx, row in enumerate(dataset):
        text = str(row["text"]).strip()
        title = str(row["title"]).strip()

        if len(text.split()) <= max_len and title:
            selected.append({
                "id": len(selected) + 1,
                "text": text,
                "title": title
            })

        if len(selected) >= max_examples:
            break

    output = {
        "dataset_type": "legal",
        "dataset_name": "eurlex",
        "dataset_lang": "eng",
        "test_cases": selected
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} examples to {output_path}")


if __name__ == "__main__":
    extract_legal_eng_examples("../../dataset/legal_eng_texts.json")
