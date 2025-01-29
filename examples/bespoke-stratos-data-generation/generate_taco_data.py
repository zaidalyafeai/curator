"""Curate data using curator on the TACO dataset."""

import argparse
import json
import resource

from datasets import load_dataset
from util.code_execution_taco import process_dataset_parallel
from util.prompt import SKY_T1_SYSTEM_PROMPT, generate_prompt
import numpy as np

from bespokelabs import curator

# Uncomment to set DeepSeek API key, or set in environment variable
# import os; os.environ['DEEPSEEK_API_KEY'] = ''


class TACOCurator(curator.LLM):
    """Curator class for processing TACO (Testing Algorithmic Coding prOblems) dataset.

    Handles prompting the LLM and parsing responses for code generation.
    """

    return_completions_object = True

    def prompt(self, problem):
        """Parse test cases and starter code from problem to create a prompt for the LLM."""
        test_case = json.loads(problem["input_output"])
        starter_code = problem["starter_code"]
        # Generate prompt text using test case, question and starter code
        prompt_text = generate_prompt(test_case, problem["question"], starter_code)
        return [{"role": "system", "content": SKY_T1_SYSTEM_PROMPT}, {"role": "user", "content": prompt_text}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["reasoning"] = response["choices"][0]["message"]["reasoning_content"]
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        return input


if __name__ == "__main__":
    # Set up command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--push_to_hub", action="store_true")
    # args.add_argument("--split", type=str, default="train")
    args.add_argument("--dataset_name", type=str, default="bespokelabs/sky-t1-taco")
    args.add_argument("--split", type=int, required=True)
    args = args.parse_args()

    # Initialize curator with DeepSeek model and parameters
    curator = TACOCurator(
        model_name="deepseek-reasoner",
        backend_params={
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30 * 60,
        },
        generation_params={
            "temp": 0.0,
            "max_tokens": 8192,
        },
    )

    # Load and filter TACO dataset based on split
    # if args.split == "train":
    #     # For training split, only use MEDIUM difficulty problems
    #     taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    # else:
    #     taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)[args.split]


    taco_dataset = load_dataset("bespokelabs/sky-t1-taco-rejection-sampled", trust_remote_code=True)['train']


    # taco_dataset = taco_dataset.add_column('input_output', taco_dataset['input_output_x'])
    taco_dataset = taco_dataset.add_column('correctness_original', taco_dataset['correctness'])

    # filter the dataset that has a function called minEatingSpeed

    # import pdb; pdb.set_trace()

    # # Generate solutions using curator
    # curated_dataset = curator(taco_dataset)

    # Push unfiltered results to hub if specified
    # if args.push_to_hub:
        # curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-unfiltered", private=True)

    # Increase file limit for parallel processing
    # import pdb; pdb.set_trace()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    # split into 10 splits
    # splits = [taco_dataset.shard(num_shards=10, index=i) for i in range(10)]

    # taco_dataset['deepseek_sol'] = 
    # drop the deepseek_solution column
    def get_deepseek_solution(x):
        try:
            return "```python\n" + json.loads(x["solutions"])[0] + "\n```"
        except:
            return "```python\n" + "\n```"


    taco_dataset = taco_dataset.shard(num_shards=5, index=args.split)
        
    taco_dataset = taco_dataset.remove_columns('deepseek_solution')
    taco_dataset = taco_dataset.map(lambda x: {"deepseek_solution": get_deepseek_solution(x)})

    # taco_dataset = taco_dataset.filter(lambda x: "maximumProfit" in x["deepseek_solution"])
    # drop rows with empty deepseek_solution
    taco_dataset = taco_dataset.filter(lambda x: x["deepseek_solution"] != "```python\n\n```")

    # Run rejection sampling to filter results
    # import pdb; pdb.set_trace()

    # drop the correctness column
    taco_dataset = taco_dataset.remove_columns('correctness')

    import pdb; pdb.set_trace()

    import time
    start_time = time.time()
    # rejection_sampled_dataset = process_dataset_parallel(taco_dataset.select(range(50,250)))
    rejection_sampled_dataset = process_dataset_parallel(taco_dataset)

    # filter the rows with correctness = False
    # rejection_sampled_dataset = rejection_sampled_dataset.filter(lambda x: x["correctness"] == False)

    import pdb; pdb.set_trace()
    # Push filtered results to hub if specified
    # if args.push_to_hub:
    # rejection_sampled_dataset.push_to_hub(f"pimpalgaonkar/code-execution-{args.split}-rejection-sampled", private=True)
    end_time = time.time()
    print(f"Time taken to process dataset: {end_time - start_time} seconds")

    import pdb; pdb.set_trace()
    print("Done")