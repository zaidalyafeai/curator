from bespokelabs import curator
from bespokelabs.curator.request_processor.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from datasets import load_dataset, Dataset
import argparse
import logging


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

    dataset = dataset.map(it_from_sharegpt, num_proc=8)
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
        "--batch",
        action="store_true",
        help="Whether to use batch processing",
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

    # python tests/test_batch.py --num_samples 10

    # if args.type == "online":
    #     api = OpenAIOnlineAPI(model="gpt-4o-mini")
    # elif args.type == "batch":
    #     api = OpenAIBatchAPI(model="gpt-4o-mini", check_interval=args.check_interval)

    # TODO(Ryan) messages as prompt_func output or if string default to user_prompt instruction
    # def prompt_func(row):
    #     messages = [
    #         {"role": "user", "content": row["instruction"]}
    #     ]
    #     return messages

    # def parse_func(row, response):
    #     row["model_response"] = response
    #     return row

    # reannotate_prompter = curator.Prompter(
    #     prompt_func=prompt_func,
    #     parse_func=parse_func,
    #     model_name="gpt-4o-mini",
    # )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    reannotate_prompter = curator.Prompter(
        prompt_func=lambda row: {"user_prompt": row["instruction"]},
        parse_func=lambda row, response: {**row, "model_response": response},
        model_name="gpt-4o-mini",
        batch=args.batch,
    )

    # To set internal variables
    request_processor = OpenAIBatchRequestProcessor(
        model="gpt-4o-mini",
        batch_size=args.batch_size,
        check_interval=args.check_interval,
    )

    reannotate_prompter._request_processor = request_processor
    reannotated_dataset = reannotate_prompter(dataset)

    # Upload dataset to Hugging Face
    print(reannotated_dataset)
    dataset_name = "mlfoundations-dev/rewrite-test-gpt-4o-mini"
    reannotated_dataset.push_to_hub(dataset_name)
    print(f"https://huggingface.co/datasets/{dataset_name}")
