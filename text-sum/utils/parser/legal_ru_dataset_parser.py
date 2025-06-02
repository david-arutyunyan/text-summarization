import os
import json
import pandas as pd


def extract_legal_ru_examples(csv_path, output_path, max_len=400, max_examples=30):
    df = pd.read_csv(csv_path)
    selected = []

    for _, row in df.iterrows():
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
        "dataset_name": "zakon",
        "dataset_lang": "ru",
        "test_cases": selected
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} examples to {output_path}")


if __name__ == "__main__":
    csv_file = "raw/zakon_ru.csv"
    output_file = "../../dataset/legal_ru_texts.json"
    extract_legal_ru_examples(csv_file, output_file)
