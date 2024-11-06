import pandas as pd
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


class InstructionResponse(BaseModel):
    response: str = Field(description="The response")


def convert_ShareGPT_to_IT_format(dataset: Dataset) -> Dataset:
    def it_from_sharegpt(sample):
        if sample["conversations"][0]["from"] == "human":
            instruction = sample["conversations"][0]["value"]
            assert sample["conversations"][1]["from"] == "gpt"
            response = sample["conversations"][1]["value"]
        elif sample["conversations"][1]["from"] == "human":
            # ignore system instructions
            instruction = sample["conversations"][1]["value"]
            assert sample["conversations"][2]["from"] == "gpt"
            response = sample["conversations"][2]["value"]
        else:
            raise ValueError("Invalid conversation format")
        return {"instruction": instruction, "original_response": response}

    dataset = dataset.map(it_from_sharegpt, num_proc=8)
    dataset = dataset.remove_columns(["conversations"])
    dataset = dataset.select_columns(["instruction", "original_response"])
    return dataset


ds = load_dataset("teknium/OpenHermes-2.5", split="train")
ds = convert_ShareGPT_to_IT_format(ds)
ds = ds.select(range(10))


def get_instruction_response(row):
    return {
        "user_prompt": f"{row['instruction']}",
    }

instruction_prompter = curator.Prompter(
    model_name="gpt-4o-mini",
    prompt_func=get_instruction_response,
    response_format=InstructionResponse,
)

ds_results = instruction_prompter(ds.to_list())

rows = []
for row, result in zip(ds, ds_results):
    rows.append({"instruction": row["instruction"], "response": result.response})

print(pd.DataFrame.from_records(rows))
