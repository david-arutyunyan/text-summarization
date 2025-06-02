import os
import json
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_amazon_reviews_dataset():
    dataset_slug = "mexwell/amazon-reviews-multi"
    local_dir = "raw"
    extracted_path = os.path.join(local_dir, "train.csv")

    os.makedirs(local_dir, exist_ok=True)

    if not os.path.exists(extracted_path):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=local_dir, unzip=True)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists locally.")

    return extracted_path


def extract_examples(csv_path, output_path, max_len=400, max_examples=30):
    df = pd.read_csv(csv_path)
    df = df[df["language"] == "en"]

    selected = []
    for _, row in df.iterrows():
        text = str(row["review_body"]).strip()
        title = str(row["review_title"]).strip()

        if len(text.split()) <= max_len and title:
            selected.append({
                "id": len(selected) + 1,
                "text": text,
                "title": title
            })
        if len(selected) >= max_examples:
            break

    output = {
        "dataset_type": "ecommerce",
        "dataset_name": "amazon",
        "dataset_lang": "eng",
        "test_cases": selected
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} samples to {output_path}")


if __name__ == "__main__":
    csv_file = download_amazon_reviews_dataset()
    extract_examples(csv_file, "../../dataset/ecommerce_eng_texts.json")
