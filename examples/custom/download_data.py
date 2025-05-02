from datasets import load_dataset
import json
import argparse
from tqdm import tqdm
import os

args = argparse.ArgumentParser()
args.add_argument("--num-examples", type=int, default=500_000)
args.add_argument("--language", type=str, default="arb_Arab")
args.add_argument("--chunk-size", type=int, default=10000)  # Number of examples per chunk

args = args.parse_args()

if args.language == "arb_Arab":
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name=args.language, split="train", streaming=True)
elif args.language == "eng_Latin":
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
else:
    raise ValueError(f"Language {args.language} not supported")

dataset = dataset.shuffle(seed=42)
dataset = dataset.take(args.num_examples)

# Create output directory if it doesn't exist
output_dir = f"fineweb-2-{args.language}-{args.num_examples}"
os.makedirs(output_dir, exist_ok=True)

# Save examples in chunks
chunk_num = 0
current_chunk = []
for i, example in enumerate(tqdm(dataset, total=args.num_examples)):
    current_chunk.append(example)
    if len(current_chunk) >= args.chunk_size or i == args.num_examples - 1:
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_num}.json")
        with open(chunk_path, "w") as f:
            json.dump(current_chunk, f)
        current_chunk = []
        chunk_num += 1

# Load the dataset from chunks
dataset = load_dataset("json", data_files=f"{output_dir}/chunk_*.json")
print(dataset)

