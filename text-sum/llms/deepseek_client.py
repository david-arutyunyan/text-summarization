import os
import json
from openai import OpenAI


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY:
    raise ValueError("Please set DEEPSEEK_API_KEY environment variable.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

DATASET_DIR = "../dataset"
PROMPT_DIR = "../prompts"
RESULTS_DIR = "../results"
MODEL_NAME = "deepseek-chat"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary(prompt_template: str, input_text: str) -> str:
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_text}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


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
                "llm_name": "deepseek",
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

            output_file_name = f"deepseek_{prompt_file.replace('_prompt.json', '')}_{dataset_file.replace('_texts.json', '')}_results.json"
            output_path = os.path.join(RESULTS_DIR, output_file_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_all()
