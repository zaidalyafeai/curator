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
import concurrent.futures
import multiprocessing
from functools import lru_cache

app = Flask(__name__)

# Global variables to store cached data and timestamp
models_data_cache = {}
last_cache_update = 0
CACHE_VALIDITY_PERIOD = 1800  # Cache validity in seconds (30 minutes)

def process_single_file(file_path):
    """Process a single model file and return its data."""
    requests_data = []
    model_name = ""
    languages = []
    dataset_id = os.path.basename(file_path)
    
    responses_file = os.path.join(file_path, "responses_0.jsonl")
    if not os.path.exists(responses_file):
        return None
        
    try:
        with open(responses_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    json_data = json.loads(line)
                    model_name = json_data["raw_request"]["model"]
                    language = json_data["generic_request"]["original_row"]["language"]
                    languages.append(language)
                    score = -1
                    generated_text = ""
                    try:
                        if "score" in json_data["parsed_response_message"][0]:
                            score_raw = json_data["parsed_response_message"][0]["score"]
                            score = float(score_raw) if isinstance(score_raw, str) else float(score_raw)
                            if score < 0 or score > 5:
                                continue
                        elif "generated_text" in json_data["parsed_response_message"][0]:
                            generated_text = json_data["parsed_response_message"][0]["generated_text"]
                    except (ValueError, TypeError):
                        continue
                    
                    reasoning = json_data["parsed_response_message"][0]["reasoning"]
                    if '"score":' in reasoning and '"reasoning":' in reasoning:
                        try:
                            if '",\n}' in reasoning:
                                reasoning = reasoning.replace('",\n}', '"}')
                            json_data = json.loads(reasoning)
                            score = json_data["score"]
                            reasoning = json_data["reasoning"]
                        except:
                            continue
                            
                    input_text = json_data["raw_request"]["messages"][0]["content"].split("The extract:")[2].strip()
                    requests_data.append({
                        "score": score,
                        "reasoning": reasoning,
                        "input_text": input_text,
                        "generated_text": generated_text
                    })
                except Exception:
                    continue
                    
        if not model_name or not requests_data:
            return None
            
        majority_language = max(set(languages), key=languages.count)
        scores = [request["score"] for request in requests_data]
        score_counts = collections.Counter(scores)
        
        score_distribution = {
            "labels": list(range(6)),
            "data": [score_counts.get(i, 0) for i in range(6)]
        }
        
        model_key = f"{model_name}-{majority_language}-{dataset_id}"
        return {
            model_key: {
                "num_requests": len(requests_data),
                "requests_data": requests_data,
                "score_distribution": score_distribution,
                "language": majority_language,
                "dataset_id": dataset_id,
                "model_name": model_name
            }
        }
    except Exception:
        return None

@lru_cache(maxsize=1)
def get_models_data(force_reload=False):
    """Process all model data and return a dictionary of models with their details.
    Uses caching and parallel processing to improve performance."""
    global models_data_cache, last_cache_update
    
    current_time = time.time()
    if not force_reload and models_data_cache and (current_time - last_cache_update) < CACHE_VALIDITY_PERIOD:
        return models_data_cache
    
    models_data = {}
    base_path = "/ibex/ai/home/alyafez/.cache/curator/"
    
    # Get list of directories to process
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) 
             if os.path.isdir(os.path.join(base_path, f))][4:6]
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_file = {executor.submit(process_single_file, file_path): file_path 
                         for file_path in files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                         total=len(files), 
                         desc="Processing model data"):
            result = future.result()
            if result:
                models_data.update(result)
    
    models_data_cache = models_data
    last_cache_update = current_time
    
    return models_data

def get_clusters(requests_data):
    import random
    """Generate cluster data for visualization."""
    clusters = []
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    for idx, request in enumerate(requests_data):
        # Use score as y-coordinate and index as x-coordinate
        # This creates a scatter plot where higher scores are higher on the y-axis
        clusters.append({
            "x": random.random()* 10,  # Use index for x-coordinate
            "y": random.random()* 10,  # Use score for y-coordinate
            "color": colors[idx % len(colors)],  # Golden angle for color distribution
            "reasoning": request["reasoning"],
            "score": request["score"]
        })
    return clusters

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
            # print("sorting by score")
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
        
        # Generate cluster data
        clusters = get_clusters(requests_data)
        
        return render_template('model_details.html', 
                              model_name=model_data['model_name'],
                              dataset_id=model_data['dataset_id'],
                              model_key=model_name,  # Pass the full model key to the template
                              model_data=model_data,
                              current_page_requests=current_page_requests,
                              page=page,
                              total_pages=total_pages,
                              sort_by=sort_by,
                              sort_order=sort_order,
                              clusters=clusters)
    return redirect(url_for('index'))

@app.route('/refresh')
def refresh_data():
    """Endpoint to force refresh the cached data."""
    get_models_data(force_reload=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 