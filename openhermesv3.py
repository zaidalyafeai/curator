import bella
from datasets import load_dataset, Dataset


def convert_ShareGPT_to_IT_format(dataset: Dataset) -> Dataset:
    def it_from_sharegpt(sample):
        if sample["conversations"][0]["from"] == "human":
            instruction = sample["conversations"][0]["value"]
            assert sample["conversations"][1]["from"] == "gpt"
            response = sample["conversations"][1]["value"]
        elif sample["conversations"][1]["from"] == "human":
            # sometimes the first message is system instructions - ignoring them here
            # specifically in OH, the system instructions are only present for airoboros2.2 and slimorca
            # airoboros2.2 provides character cards or "you are a trivia AI"
            # slimorca provides CoT instructions a la "you are a helpful assistant and explain your steps"
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

ds = bella.completions(
    dataset=ds,
    prompter=bella.Prompter(
        user_prompt="{{instruction}}",
        model_name="gpt-4o-mini",
    ),
    output_column="model_response",
)

print(ds)
