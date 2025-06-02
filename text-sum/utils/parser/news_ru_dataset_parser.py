import os
import pandas as pd
import json
from kaggle.api.kaggle_api_extended import KaggleApi


def download_lenta_dataset():
    dataset_slug = "yutkin/corpus-of-russian-news-articles-from-lenta"
    local_csv = "raw/lenta-ru-news.csv"

    os.makedirs("raw", exist_ok=True)

    if not os.path.exists(local_csv):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path="raw", unzip=True)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists locally.")

    return local_csv


def extract_examples(csv_path, output_path, max_len=400, max_examples=30):
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
        "dataset_type": "news",
        "dataset_name": "lenta",
        "dataset_lang": "ru",
        "test_cases": selected
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} samples to {output_path}")


if __name__ == "__main__":
    csv_file = download_lenta_dataset()
    extract_examples(csv_file, "../../dataset/news_ru_texts.json")
