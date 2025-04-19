import json
import os

def process_json(response):
    response = response.replace("```json", "").strip()
    response = response.replace("```", "").strip()

    return json.loads(response)
base_path = "/ibex/ai/home/alyafez/.cache/curator/"

for file in os.listdir(base_path):
    if os.path.isdir(f"{base_path}/{file}"):
        if os.path.exists(f"{base_path}/{file}/responses_0.jsonl"):
            # For JSONL files (JSON Lines format - one JSON object per line)
            requests = []
            print(f"{base_path}/{file}")
            model_name = ""

            with open(f"{base_path}/{file}/responses_0.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            json_data = json.loads(line)
                            model_name = json_data["raw_request"]["model"]
                            score = json_data["parsed_response_message"][0]["score"]
                            reasoning = json_data["parsed_response_message"][0]["reasoning"]
                            input_text = json_data["raw_request"]["messages"][0]["content"].split("The extract:")[2].strip()
                            
                            requests.append(json_data)
                        except Exception as e:
                            print(e)

            # plot by score
            import matplotlib.pyplot as plt
            num_requests = len(requests)
            print("processed ", num_requests, " requests")
            scores = [request["parsed_response_message"][0]["score"] for request in requests]
            import collections
            score_counts = collections.Counter(scores)
            print(score_counts)
            plt.bar(score_counts.keys(), score_counts.values())
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.title(f"Score Distribution for {model_name} with {num_requests} requests")
            plt.xlim(0, 5)
            plt.savefig(f"score_{model_name}_{num_requests}.png")
            plt.close()