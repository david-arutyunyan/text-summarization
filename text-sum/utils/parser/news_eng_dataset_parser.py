import json
from datasets import load_dataset


def load_and_save_cnn_dailymail(output_path, max_len=400, max_examples=30):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    selected_cases = []

    for item in dataset:
        word_count = len(item["article"].split())
        if word_count <= max_len:
            selected_cases.append({
                "id": len(selected_cases) + 1,
                "text": item["article"].strip(),
                "title": item["highlights"].strip()
            })
        if len(selected_cases) == max_examples:
            break

    output_data = {
        "dataset_type": "news",
        "dataset_name": "cnn",
        "dataset_lang": "eng",
        "test_cases": selected_cases
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected_cases)} examples to {output_path}")


if __name__ == "__main__":
    load_and_save_cnn_dailymail("../../dataset/news_eng_texts.json")