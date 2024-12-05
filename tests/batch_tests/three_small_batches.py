from bespokelabs.curator import Prompter
from datasets import Dataset
import logging
import argparse


def main(args):
    if args.log_level is not None:
        logger = logging.getLogger("bespokelabs.curator")
        logger.setLevel(args.log_level)

    dataset = Dataset.from_dict({"prompt": ["just say 'hi'"] * 3})

    prompter = Prompter(
        prompt_func=lambda row: row["prompt"],
        model_name="gpt-4o-mini",
        response_format=None,
        batch=True,
        batch_size=1,
        batch_check_interval=10,
        batch_cancel=args.cancel,
    )

    dataset = prompter(dataset)
    print(dataset.to_pandas())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prompter with optional cancellation.")
    parser.add_argument("--cancel", action="store_true", default=False, help="Cancel the batches")
    parser.add_argument(
        "--log-level",
        type=lambda x: getattr(logging, x.upper()),
        default=None,
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()
    main(args)
