import requests

url = "http://localhost:8787/v1/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "google/gemma-3-27b-it",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0,
    "max_model_len": 27872,  # Setting to the maximum supported KV cache size
    "gpu_memory_utilization": 0.9  # Setting to 90% utilization
}

response = requests.post(url, headers=headers, json=data)
print(response.json()) 