import logging

from datasets import load_dataset

from bespokelabs import curator

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


def convert_row(row: dict) -> dict:
    conversation = row["conversations"]
    instruction = next((item["value"] for item in conversation if item["from"] == "human"), None)
    response = next((item["value"] for item in conversation if item["from"] == "gpt"), None)
    return {"instruction": instruction, "original_response": response}


def prompt_func(row):
    return row["instruction"]


def parse_func(row, response):
    instruction = row["instruction"]
    return {"instruction": instruction, "new_response": response}


distill_prompter = curator.LLM(
    prompt_func=prompt_func,
    parse_func=parse_func,
    model_name="claude-3-5-sonnet-20241022",
    batch=True,
    batch_size=100,
)

dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
dataset = dataset.take(500)
dataset = dataset.map(convert_row)
dataset = dataset.select_columns(["instruction", "original_response"])
distilled_dataset = distill_prompter(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
