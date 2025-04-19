from datasets import load_dataset
import json
import argparse
from tqdm import tqdm
args = argparse.ArgumentParser()
args.add_argument("--num-examples", type=int, default=1000)
args.add_argument("--language", type=str, default="arb_Arab")

args = args.parse_args()

if args.language == "arb_Arab":
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name=args.language, split="train", streaming=True)
elif args.language == "eng_Latin":
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
else:
    raise ValueError(f"Language {args.language} not supported")

dataset = dataset.take(args.num_examples)

examples = []
for example in tqdm(dataset):
    examples.append(example)

# save the examples to a json file
with open(f"fineweb-2-{args.language}-{args.num_examples}.json", "w") as f:
    json.dump(examples, f)


# load the examples from the json file using datasets
dataset = load_dataset("json", data_files=f"fineweb-2-{args.language}-{args.num_examples}.json")
print(dataset)

