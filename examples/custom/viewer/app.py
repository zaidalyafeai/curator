from flask import Flask, render_template, request, redirect, url_for
import json
import os
import collections
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
from tqdm import tqdm

app = Flask(__name__)

# Global variables to store cached data and timestamp
models_data_cache = {}
last_cache_update = 0
CACHE_VALIDITY_PERIOD = 300  # Cache validity in seconds (5 minutes)

def get_models_data(force_reload=False):
    """Process all model data and return a dictionary of models with their details.
    Uses caching to avoid reloading data on every request."""
    global models_data_cache, last_cache_update
    
    current_time = time.time()
    # Check if cache is valid and not empty
    if not force_reload and models_data_cache and (current_time - last_cache_update) < CACHE_VALIDITY_PERIOD:
        return models_data_cache
    
    # Cache is invalid or empty, reload data
    models_data = {}
    base_path = "/ibex/ai/home/alyafez/.cache/curator/"
    
    # Get list of files first to show progress
    files = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for file in tqdm(files, desc="Processing model data"):
        file_path = os.path.join(base_path, file)
        if os.path.isdir(file_path):
            responses_file = os.path.join(file_path, "responses_0.jsonl")
            if os.path.exists(responses_file):
                requests_data = []
                model_name = ""
                langauges = []
                dataset_id = file  # Use the directory name as dataset identifier
                with open(responses_file, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                json_data = json.loads(line)
                                if ", }" in line:
                                    json_data = json_data.replace(", }", "}")
                                model_name = json_data["raw_request"]["model"]
                                language = json_data["generic_request"]["original_row"]["language"]
                                langauges.append(language)
                                # Properly parse and convert the score to a numeric value
                                try:
                                    score_raw = json_data["parsed_response_message"][0]["score"]
                                    # If score is a string, try to convert to float
                                    if isinstance(score_raw, str):
                                        score = float(score_raw)
                                    else:
                                        score = float(score_raw)
                                    # Ensure score is within expected range (0-5)
                                    if score < 0 or score > 5:
                                        # print(f"Warning: Score out of expected range: {score}")
                                        score = 0
                                except (ValueError, TypeError):
                                    # print(f"Error converting score to number: {score_raw}")
                                    score = 0
                                
                                reasoning = json_data["parsed_response_message"][0]["reasoning"]
                                if '"score":' in reasoning and '"reasoning":' in reasoning:                                    
                                    try:
                                        if '",\n}' in reasoning:
                                            reasoning = reasoning.replace('",\n}', '"}')
                                        json_data = json.loads(reasoning)
                                        score = json_data["score"]
                                        reasoning = json_data["reasoning"]
                                    except:
                                        # print("error")
                                        # print(reasoning)
                                        continue
                                input_text = json_data["raw_request"]["messages"][0]["content"].split("The extract:")[2].strip()
                                
                                requests_data.append({
                                    "score": score,
                                    "reasoning": reasoning,
                                    "input_text": input_text
                                })
                            except Exception as e:
                                # print(f"Error processing line: {e}")
                                pass

                majority_langauge = max(set(langauges), key=langauges.count)
                if model_name and requests_data:
                    # Calculate score distribution
                    scores = [request["score"] for request in requests_data]
                    score_counts = collections.Counter(scores)
                    
                    # Create score distribution data
                    score_distribution = {
                        "labels": list(range(6)),  # Scores from 0 to 5
                        "data": [score_counts.get(i, 0) for i in range(6)]
                    }

                    # Create a unique key combining model name, language, and dataset ID
                    model_key = f"{model_name}-{majority_langauge}-{dataset_id}"
                    models_data[model_key] = {
                        "num_requests": len(requests_data),
                        "requests_data": requests_data,
                        "score_distribution": score_distribution,
                        "language": majority_langauge,
                        "dataset_id": dataset_id,
                        "model_name": model_name
                    }
    
    # Update cache and timestamp
    models_data_cache = models_data
    last_cache_update = current_time
    
    return models_data

def generate_score_graph(requests_data, model_name):
    """Generate a base64 encoded image of the score distribution."""
    scores = [request["score"] for request in requests_data]
    score_counts = collections.Counter(scores)
    
    plt.figure(figsize=(8, 6))
    plt.bar(score_counts.keys(), score_counts.values())
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Score Distribution for {model_name} with {len(requests_data)} requests")
    plt.xlim(-1, 6)
    
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    """Home page showing all models and their request counts."""
    models_data = get_models_data()
    return render_template('index.html', models_data=models_data)

@app.route('/model/<model_name>')
def model_details(model_name):
    """Detail page for a specific model."""
    models_data = get_models_data()
    if model_name in models_data:
        model_data = models_data[model_name]
        
        # Get page parameter from the query string, default to page 1
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        # Get sort parameter from query string, default to none
        sort_by = request.args.get('sort', None)
        sort_order = request.args.get('order', 'desc')
        
        # Create a deep copy of the requests data to avoid modifying the cached data
        requests_data = model_data['requests_data'].copy()
        # Ensure scores are properly converted to numbers before sorting
        if sort_by == 'score':
            print("sorting by score")
            # Make sure scores are treated as numeric values
            for req in requests_data:
                if isinstance(req['score'], str):
                    try:
                        req['score'] = float(req['score'])
                    except ValueError:
                        # Handle non-numeric scores - set to 0 or another default
                        req['score'] = 0
            
            # Sort with proper numeric comparison
            requests_data.sort(key=lambda x: float(x['score']), reverse=(sort_order == 'desc'))
        
        # Calculate pagination indices
        total_requests = len(requests_data)
        total_pages = (total_requests + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_requests)
        
        # Get the subset of requests for current page
        current_page_requests = requests_data[start_idx:end_idx]
        
        return render_template('model_details.html', 
                              model_name=model_data['model_name'],
                              dataset_id=model_data['dataset_id'],
                              model_key=model_name,  # Pass the full model key to the template
                              model_data=model_data,
                              current_page_requests=current_page_requests,
                              page=page,
                              total_pages=total_pages,
                              sort_by=sort_by,
                              sort_order=sort_order)
    return redirect(url_for('index'))

@app.route('/refresh')
def refresh_data():
    """Endpoint to force refresh the cached data."""
    get_models_data(force_reload=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 