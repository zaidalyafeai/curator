import os
from datasets import Dataset, load_dataset
from bespokelabs import curator
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pydantic import BaseModel
import requests
import time
from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import NewConnectionError, MaxRetryError
from openai import OpenAI
import argparse
import json

class Score(BaseModel):
    score: int
    reasoning: str

load_dotenv("../../.env")

prompt = """Below is an extract from a web page. You need to generate a high educational
content based on the text below that could be useful in an educational setting 
for teaching from primary school to grade school levels. The generated text must be in 
the same language as the extract. It should expand the extract to be more informative and 
detailed.
The extract: <EXAMPLE>.
After examining the extract:
- Generated text: "generated_text: <generated_text>"
- Reason why the generated text has educational value: "reasoning: <reasoning>"
Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python. 
It should be in the following schema, don't add any extra text or json headers: 
{
    "reasoning": <reasoning>,
    "generated_text": <generated_text>,
}
"""


import re
def process_json(response):
    response = response.replace("```json", "").strip()
    response = response.replace("```", "").strip()
    return response

class Prompter(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"{prompt}\nThe extract: {input['text'][:10_000]}"

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        response = process_json(response)
        try:
            response = json.loads(response)
            return {
                "generated_text": response["generated_text"],
                "reasoning": response["reasoning"],
            }
        except Exception as e:
            return {
                "generated_text": "",
                "reasoning": response
            }


def main():

    
    
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="vllm")
    args.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    args.add_argument("--max-requests-per-minute", type=int, default=100)
    args.add_argument("--max-retries", type=int, default=50)
    args.add_argument("--language", type=str, default="arb_Arab")
    args.add_argument("--num-examples", type=int, default=10000)
    args = args.parse_args()

    fw  = load_dataset("json", data_files=f"fineweb-2-{args.language}-{args.num_examples}.json")["train"]
    fw = fw.select(range(args.num_examples))
    

    HOST = "localhost"
    PORT = 8787
    if args.mode == "vllm-online":
        model_name = args.model
        model_name = f"hosted_vllm/{model_name}"  # Use the hosted_vllm backend

        MAX_RETRIES = 30  # 5 minutes with 10 second intervals
        RETRY_INTERVAL = 10
        os.environ["HOSTED_VLLM_API_KEY"] = "EMPTY"
        
        MAX_RETRIES = 30  # 5 minutes with 10 second intervals
        RETRY_INTERVAL = 10

        def check_server_status(host, port):
            url = f"http://{host}:{port}/v1"
            try:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url=url,
                )
                print("running inference")
                chat_response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is ai?"},
                    ]
                )
                print("inference done")
                print(chat_response)
                return True
            except ConnectionError as e:
                # This catches both ConnectionRefusedError and other connection issues
                if isinstance(e.args[0], MaxRetryError) and isinstance(e.args[0].reason, NewConnectionError):
                    # This is the specific case of connection refused
                    return False
                print(f"Connection error: {e}")
                return False
            except RequestException as e:
                print(f"Request error: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False

        retry_count = 0
        while not check_server_status(HOST, PORT):
        #while False:
            if retry_count >= MAX_RETRIES:
                print("Maximum retries reached. Server failed to start.")
                exit(1)
            print(f"Waiting for the server to start... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_INTERVAL)
            retry_count += 1
        print("Server is running")
        llm = Prompter(
            model_name=args.model,
            backend="openai",
            backend_params={
            "base_url": f"http://{HOST}:{PORT}/v1",
            "api_key" : "EMPTY"
            },
            #response_format=Score,
    )
    elif args.mode == "vllm-offline":

        llm = Prompter(
            model_name=args.model,
            backend="vllm",
            backend_params={ 
                "tensor_parallel_size": 4, # Adjust based on GPU count 
                "gpu_memory_utilization": 0.95,
                "max_model_length": 16384,
            },
            response_format=Score,
        )
    else:
        model_name = f"openrouter/qwen/qwen-2.5-72b-instruct"
        llm = Prompter(
        model_name=model_name,
        backend_params={
            "max_requests_per_minute": args.max_requests_per_minute,
            "request_timeout": 30 * 60,
        },
        #response_format=Score,
    )
    

    

    responses = llm(fw)

    # convert to pandas
    df = responses.to_pandas()


if __name__ == "__main__":
    main()
