from bespokelabs import curator
from datasets import load_dataset

dataset = load_dataset("allenai/WildChat", split="train")
dataset = dataset.select(range(3_000))


def prompt_func(row):
    return row["conversation"][0]["content"]


def parse_func(row, response):
    instruction = row["conversation"][0]["content"]
    return {"instruction": instruction, "new_response": response}


distill_prompter = curator.Prompter(
    prompt_func=prompt_func, parse_func=parse_func, model_name="gpt-4o-mini", batch=True
)

distilled_dataset = distill_prompter(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
