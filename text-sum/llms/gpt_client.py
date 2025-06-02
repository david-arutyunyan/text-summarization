import os
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

DATASET_DIR = "../dataset"
PROMPT_DIR = "../prompts"
RESULTS_DIR = "../results"

AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION = "2024-03-01-preview"

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION,
    temperature=0.2,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary(prompt_template: str, input_text: str) -> str:
    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=input_text)
    ]
    response = llm.invoke(messages)
    return response.content.strip()


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
                "llm_name": AZURE_DEPLOYMENT_NAME,
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

            output_file_name = f"gpt_{prompt_file.replace('_prompt.json', '')}_{dataset_file.replace('_texts.json', '')}_results.json"
            output_path = os.path.join(RESULTS_DIR, output_file_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_all()
