from bespokelabs.curator import Prompter
from datasets import Dataset
import logging

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

n_prompts = 10_000
dataset = Dataset.from_dict({"prompt": ["write me a poem"] * n_prompts})

prompter = Prompter(
    prompt_func=lambda row: row["prompt"],
    model_name="gpt-4o-mini",
    response_format=None,
    batch=True,
    batch_size=1,
)

dataset = prompter(dataset)
