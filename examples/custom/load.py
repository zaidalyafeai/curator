from datasets import load_dataset, Dataset
from viewer.app import process_single_file
import os

base_path = "/ibex/ai/home/alyafez/.cache/curator/"
file_name = os.listdir(base_path)[2]
file_path = os.path.join(base_path, file_name)
print(file_path)
data = process_single_file(file_path, max_num_requests=None)
key = list(data.keys())[0]
print(len(data[key]['requests_data']))
dataset = data[key]['requests_data']

# only keep input_text and score
dataset = [{"input_text": x['input_text'], "score": x['score'], "reasoning": x['reasoning']} for x in dataset]
dataset = Dataset.from_list(dataset)
print(dataset)
dataset.save_to_disk("annotated_dataset")
