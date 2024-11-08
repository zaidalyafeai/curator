from bespokelabs import curator
from bespokelabs.curator import OpenAIBatchRequestProcessor
from datasets import load_dataset, Dataset
import argparse


def convert_ShareGPT_to_IT_format(dataset: Dataset) -> Dataset:
    def it_from_sharegpt(sample):
        if sample["conversations"][0]["from"] == "human":
            instruction = sample["conversations"][0]["value"]
            assert sample["conversations"][1]["from"] == "gpt"
            response = sample["conversations"][1]["value"]
        elif sample["conversations"][1]["from"] == "human":
            instruction = sample["conversations"][1]["value"]
            assert sample["conversations"][2]["from"] == "gpt"
            response = sample["conversations"][2]["value"]
        else:
            raise ValueError("Invalid conversation format")
        return {"instruction": instruction, "original_response": response}

    dataset = dataset.map(it_from_sharegpt)
    dataset = dataset.remove_columns(["conversations"])
    dataset = dataset.select_columns(["instruction", "original_response"])
    return dataset


def load_ShareGPT_dataset_as_IT(dataset_name: str, truncate: int = None) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if truncate is not None:
        dataset = dataset.select(range(truncate))
    return convert_ShareGPT_to_IT_format(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working_dir",
        type=str,
        required=True,
        help="Where requests, responses, and dataset will be locally written to save intermediate results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="The number of samples to use from the dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="The number of samples to use per batch",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="online",
        help="The type of API to use",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=10,
        help="The interval (in seconds) to check the status of the batch",
    )
    args = parser.parse_args()

    # Load the dataset to instruction, response columns
    dataset = load_ShareGPT_dataset_as_IT("teknium/OpenHermes-2.5")
    print(dataset)

    dataset = dataset.select(range(args.num_samples))

    # if args.type == "online":
    #     api = OpenAIOnlineAPI(model="gpt-4o-mini")
    # elif args.type == "batch":
    #     api = OpenAIBatchAPI(model="gpt-4o-mini", check_interval=args.check_interval)
    reannotate_prompter = curator.Prompter(
        prompt_func=lambda row: {"user_prompt": row["instruction"]},
        parse_func=lambda row, response: {**row, "model_response": response},
        model_name="gpt-4o-mini",
    )

    request_processor = OpenAIBatchRequestProcessor(
        model="gpt-4o-mini",
        batch_size=args.batch_size,
        check_interval=args.check_interval,
    )

    reannotated_dataset = reannotate_prompter(dataset, request_processor)

    dataset = reannotated_dataset.to_huggingface()

    # Upload dataset to Hugging Face
    print(dataset)
    dataset_name = "mlfoundations-dev/rewrite-test-gpt-4o-mini"
    dataset.push_to_hub(dataset_name)
    print(f"https://huggingface.co/datasets/{dataset_name}")
