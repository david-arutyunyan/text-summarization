import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATASET_DIR = "../dataset"
PROMPT_DIR = "../prompts"
RESULTS_DIR = "../results"

MODEL_NAME = "google/mt5-base"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary(prompt_template: str, input_text: str) -> str:
    full_prompt = f"{prompt_template}\n\n{input_text}"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True).to(device)
    output_tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text.strip()


def run_all():
    for dataset_file in os.listdir(DATASET_DIR):
        if not dataset_file.endswith(".json"):
            continue
        dataset_path = os.path.join(DATASET_DIR, dataset_file)
        dataset = load_json(dataset_path)

        for prompt_file in os.listdir(PROMPT_DIR):
            if not prompt_file.endswith(".json"):
                continue
            prompt_path = os.path.join(PROMPT_DIR, prompt_file)
            prompt = load_json(prompt_path)["prompt"]

            results = {
                "llm_name": "mt5",
                "prompt_method": prompt_file.replace("_prompt.json", ""),
                "dataset_type": dataset["dataset_type"],
                "dataset_name": dataset["dataset_name"],
                "dataset_lang": dataset["dataset_lang"],
                "results": []
            }

            for test_case in dataset["test_cases"]:
                summary = generate_summary(prompt, test_case["text"])
                results["results"].append({
                    "test_case_id": test_case["id"],
                    "text": test_case["text"],
                    "generated_title": summary
                })

            output_file_name = f"mt5_{prompt_file.replace('_prompt.json', '')}_{dataset_file.replace('_texts.json', '')}_results.json"
            output_path = os.path.join(RESULTS_DIR, output_file_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_all()
