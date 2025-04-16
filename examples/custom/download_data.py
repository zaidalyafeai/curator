from datasets import load_dataset
import json
import argparse
from tqdm import tqdm
args = argparse.ArgumentParser()
args.add_argument("--num-examples", type=int, default=1000)
args = args.parse_args()

dataset = load_dataset("HuggingFaceFW/fineweb-2", name="arb_Arab", split="train", streaming=True)

dataset = dataset.take(args.num_examples)

examples = []
for example in tqdm(dataset):
    examples.append(example)

# save the examples to a json file
with open(f"fineweb-2-arb-Arab-{args.num_examples}.json", "w") as f:
    json.dump(examples, f)


# load the examples from the json file using datasets
dataset = load_dataset("json", data_files=f"fineweb-2-arb-Arab-{args.num_examples}.json")
print(dataset)

