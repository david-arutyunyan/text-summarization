import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
RESULTS_DIR = "../results"
PROMPT_PATH = "../prompts/llm_evaluator_prompt/llama_evaluator_prompt.json"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "llm_evaluator_results")
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(template, text, summary):
    return template.replace("{text}", text).replace("{summary}", summary)


def parse_scores(output_text):
    lines = output_text.strip().splitlines()
    scores = []
    for line in lines:
        line = line.strip()
        if line.isdigit():
            scores.append(int(line))
    return scores if len(scores) == 3 else [3, 3, 3]


def evaluate():
    prompt_template = load_json(PROMPT_PATH)["prompt"]

    for file in os.listdir(RESULTS_DIR):
        if not file.endswith("_results.json") or "llm_evaluator_results" in file:
            continue

        input_path = os.path.join(RESULTS_DIR, file)
        result_data = load_json(input_path)

        evaluated = {
            "llm_name": result_data["llm_name"],
            "prompt_method": result_data["prompt_method"],
            "dataset_type": result_data["dataset_type"],
            "dataset_name": result_data["dataset_name"],
            "dataset_lang": result_data["dataset_lang"],
            "results": []
        }

        for item in result_data["results"]:
            prompt = build_prompt(prompt_template, item["text"], item["generated_title"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)

            fc, rel, flu = parse_scores(decoded)
            evaluated["results"].append({
                "test_case_id": item["test_case_id"],
                "text": item["text"],
                "generated_title": item["generated_title"],
                "factual_correctness": fc,
                "relevance": rel,
                "fluency": flu
            })

        output_name = file.replace("_results.json", "_evaluated_results.json")
        output_path = os.path.join(OUTPUT_DIR, output_name)
        save_json(output_path, evaluated)
        print(f"Evaluated: {output_name}")


if __name__ == "__main__":
    evaluate()
