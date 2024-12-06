from bespokelabs.curator import Prompter
from datasets import Dataset
import logging
import argparse

# python tests/speed_test.py --log-level DEBUG --model claude-3-5-haiku-20241022


def main(args):
    if args.log_level is not None:
        logger = logging.getLogger("bespokelabs.curator")
        logger.setLevel(args.log_level)

    dataset = Dataset.from_dict({"prompt": ["write me a poem"] * args.n_requests})

    prompter = Prompter(
        prompt_func=lambda row: row["prompt"],
        model_name=args.model,
    )

    dataset = prompter(dataset, batch_cancel=args.cancel)
    print(dataset.to_pandas())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple batch test bed")
    parser.add_argument("--cancel", action="store_true", default=False, help="Cancel the batches")
    parser.add_argument(
        "--n-requests", type=int, help="Number of requests to process", default=100_000
    )
    parser.add_argument(
        "--log-level",
        type=lambda x: getattr(logging, x.upper()),
        default=None,
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gemini/gemini-1.5-flash")
    args = parser.parse_args()
    main(args)
