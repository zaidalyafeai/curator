from bespokelabs.curator import Prompter
from datasets import Dataset
import logging

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

dataset = Dataset.from_dict({"prompt": ["write me a poem"] * 60_000})

prompter = Prompter(
    prompt_func=lambda row: row["prompt"],
    model_name="gpt-4o-mini",
    response_format=None,
    batch=True,
)

dataset = prompter(dataset)
print(dataset.to_pandas())
