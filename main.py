####################################
# main.py
# Replit: Continual Mining + Leaderboard
# FIXED: Convert ObservedList to plain list to avoid concat errors
####################################

import requests
import random
import torch
import time
import threading
import os
from os import getenv
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, render_template_string
from replit import db
from datetime import datetime

###############################
# CONFIGURATIONS
###############################
POPULATION_SIZE = 10       # Lower for fewer requests
NUM_GENERATIONS_PER_RUN = 2
KEEP_RATIO = 0.5
MUTATION_RATE = 0.02
PAGE_ID_LENGTH = 32
REQUEST_TIMEOUT = 10       # seconds
MAX_TEXT_LENGTH = 1000
MAX_EMBED_TOKENS = 256

# After each GA "run," how long to sleep (seconds) before next run
SLEEP_BETWEEN_RUNS = 5

# Replit DB KEY where we store top hits
TOP_HITS_DB_KEY = "top_hits"
MAX_STORED_HITS = 20       # keep top 20 in the DB

###############################
# MODEL INITIALIZATION
###############################
model = None
tokenizer = None

def load_model_background():
    global model, tokenizer
    print("Loading transformer model in background...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    print("Model loaded.")

###############################
# FLASK APP SETUP
###############################
app = Flask(__name__)

# Add a health check endpoint
@app.route("/health")
def health():
    return "OK", 200

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Library of Babel - Evolutionary Semantic Miner</title>
</head>
<body style="background: #000; color: #0f0; font-family: monospace; margin:0; padding:20px;">
  <h1>Library of Babel</h1>
  <p>This application explores the Library of Babel using genetic algorithms and machine learning. 

The Library of Babel is a Borges inspried website (libraryofbabel.info) that generates pseudo-random text pages based on unique page IDs. Project Tommyknockers attempts to mine the Library of Babel for meaningful text within this vast space of randomness by:
  <ul>
    <li>Generating a random population of page IDs.</li>
    <li>Retrieving and scoring each page’s coherence using a language model.</li>
    <li>Keeping the best-performing pages, mutating them, and repeating.</li>
  </ul>
 This is essentially an AI-powered exploration tool trying to find needles (coherent text) in an astronomically large haystack (the Library of Babel’s random text space). The prize? Every book, poem or idea that ever has been or ever will be written lies within the Library of Babel...waiting for a clever librarian.</p>
  <p>Top discoveries appear in a running scoreboard. Visit the <a href="/leaderboard" style="color: #0ff;">the leaderboard</a>  to see the current best hits discovered so far.</p>
</body>
</html>
"""

LEADERBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Library of Babel - Leaderboard</title>
  <style>
    body {
      background: #000;
      color: #0f0;
      font-family: monospace;
      margin:0;
      padding:20px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      max-width: 900px;
      margin: 0 auto;
    }
    th, td {
      border: 1px solid #0f0;
      padding: 8px;
      text-align: left;
    }
    a {
      color: #0ff;
    }
    h1 {
      text-align: center;
    }
    .metrics {
      font-size: 0.9em;
      color: #0ff;
    }
  </style>
</head>
<body>
  <h1>Top Hits Leaderboard</h1>
  <p style="text-align:center;">Showing up to {{ max_stored }} best pages discovered so far.</p>
  <table>
    <tr>
      <th>Rank</th>
      <th>Page ID</th>
      <th>Score</th>
      <th>Snippet</th>
      <th>Metrics</th>
      <th>Discovered</th>
    </tr>
    {% for item in hits %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><a href="https://libraryofbabel.info/book.cgi?{{ item.page_id }}" target="_blank">{{ item.page_id[:8] }}...</a></td>
      <td>{{ item.score | round(3) }}</td>
      <td>{{ item.snippet }}</td>
      <td class="metrics">
        <pre>{{ item.metrics }}</pre>
      </td>
      <td>{{ item.timestamp }}</td>
    </tr>
    {% endfor %}
  </table>
  <p style="text-align:center;margin-top:20px;"><a href="/">Back Home</a></p>
</body>
</html>
"""

@app.route("/")
def home():
    return HOME_HTML

@app.route("/leaderboard")
def leaderboard():
    # Get top hits from DB
    hits = db.get(TOP_HITS_DB_KEY, [])
    # Convert ObservedList to normal list
    hits = list(hits)
    # Sort by score descending
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)
    return render_template_string(LEADERBOARD_TEMPLATE, hits=hits_sorted, max_stored=MAX_STORED_HITS)

###############################
# GA & SCRAPING LOGIC
###############################
def get_page_text(page_id: str) -> str:
    url = f"https://libraryofbabel.info/book.cgi?{page_id}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            page_div = soup.find("div", {"id": "page"})
            if page_div:
                return page_div.get_text()
    except:
        pass
    return ""

def get_semantic_score_and_embedding(text: str):
    """
    Returns a (score, embedding) tuple.
    'score' = embedding norm as a rough measure of coherence.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        return 0.0, None

    text = text.strip()
    if not text:
        return 0.0, None
    text = text[:MAX_TEXT_LENGTH]  # limit length
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_EMBED_TOKENS)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    sentence_embedding = token_embeddings.mean(dim=1).squeeze()
    norm_val = torch.norm(sentence_embedding).item()
    return float(norm_val), sentence_embedding

def interpret_text(snippet: str, embedding_norm: float):
    """
    Provide interpretability metrics as a dict.
    """
    snippet = snippet.strip()
    if not snippet:
        return {
            "Token count": 0,
            "Avg word length": 0.0,
            "Alphabetic ratio": 0.0,
            "Repeated word ratio": 0.0,
            "Embedding norm": embedding_norm
        }

    tokens = snippet.split()
    count = len(tokens)
    cleaned = [t.strip(",.!?;:\"'()[]{}") for t in tokens]
    if count == 0:
        return {
            "Token count": 0,
            "Avg word length": 0.0,
            "Alphabetic ratio": 0.0,
            "Repeated word ratio": 0.0,
            "Embedding norm": embedding_norm
        }

    avg_len = sum(len(t) for t in cleaned) / count
    alpha_tokens = [t for t in cleaned if t.isalpha()]
    alpha_ratio = len(alpha_tokens) / count if count else 0
    unique_tokens = set(cleaned)
    repeated_count = count - len(unique_tokens)
    repeated_ratio = repeated_count / count if count else 0

    return {
        "Token count": count,
        "Avg word length": round(avg_len, 2),
        "Alphabetic ratio": round(alpha_ratio, 2),
        "Repeated word ratio": round(repeated_ratio, 2),
        "Embedding norm": round(embedding_norm, 3)
    }

def random_page_id(length=PAGE_ID_LENGTH):
    return ''.join(random.choice('0123456789abcdef') for _ in range(length))

def mutate(pid, mutation_rate):
    chars = list(pid)
    for i in range(len(chars)):
        if random.random() < mutation_rate:
            possible_digits = [d for d in '0123456789abcdef' if d != chars[i]]
            chars[i] = random.choice(possible_digits)
    return ''.join(chars)

def crossover(p1, p2):
    half = len(p1) // 2
    return p1[:half] + p2[half:]

def evaluate_population(pop):
    results = []
    for pid in pop:
        full_text = get_page_text(pid)
        score, _ = get_semantic_score_and_embedding(full_text)
        snippet = full_text[:300].replace("\n", " ")
        metrics = interpret_text(snippet, score)
        results.append({
            "page_id": pid,
            "score": score,
            "snippet": snippet,
            "metrics": metrics
        })
    return results

def select_parents(evaluated, keep_ratio):
    sorted_eval = sorted(evaluated, key=lambda x: x["score"], reverse=True)
    k = max(1, int(len(sorted_eval) * keep_ratio))
    return sorted_eval[:k]

def breed_population(parents, new_size):
    parent_ids = [p["page_id"] for p in parents]
    new_pop = []
    while len(new_pop) < new_size:
        p1 = random.choice(parent_ids)
        p2 = random.choice(parent_ids)
        child = crossover(p1, p2)
        new_pop.append(child)
    return new_pop

###############################
# SAVE/LOAD to Replit DB
###############################
def load_top_hits():
    """
    Returns a list of dicts: [{page_id, score, snippet, metrics, timestamp}, ...]
    We convert ObservedList -> normal list to avoid concat errors.
    """
    data = db.get(TOP_HITS_DB_KEY, [])
    return list(data)  # convert to plain Python list

def save_top_hits(hits_list):
    db[TOP_HITS_DB_KEY] = hits_list

def try_update_leaderboard(candidates):
    """
    'candidates' is a list of GA-evaluated items. We check if any have a score
    better than the lowest in the current DB, or if the DB is not full.
    We'll keep only top N by score.
    """
    top_hits = load_top_hits()  # now a normal list
    combined = top_hits + candidates
    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)
    top_trimmed = combined_sorted[:MAX_STORED_HITS]

    # Add a timestamp to new items
    existing_ids = set(hit["page_id"] for hit in top_hits)
    now_str = datetime.utcnow().isoformat()
    for item in top_trimmed:
        if "timestamp" not in item or not item["timestamp"]:
            if item["page_id"] not in existing_ids:
                item["timestamp"] = now_str

    save_top_hits(top_trimmed)

###############################
# BACKGROUND WORKER
###############################
class GAWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        """Continually run small GA searches, store top hits in DB."""
        while self.running:
            # Clear memory before each run
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 1. Initialize population
            population = [random_page_id() for _ in range(POPULATION_SIZE)]
            # 2. For a few generations, refine
            for _ in range(NUM_GENERATIONS_PER_RUN):
                evaluated = evaluate_population(population)
                parents = select_parents(evaluated, KEEP_RATIO)
                new_pop = breed_population(parents, POPULATION_SIZE)
                mutated = [mutate(pid, MUTATION_RATE) for pid in new_pop]
                population = mutated

            # Evaluate final pop
            final_evaluated = evaluate_population(population)
            # Update leaderboard if any are good
            try_update_leaderboard(final_evaluated)

            # Short sleep to avoid spamming requests
            time.sleep(SLEEP_BETWEEN_RUNS)

    def stop(self):
        self.running = False

###############################
# LAUNCH BACKGROUND PROCESSES
###############################
# Start model loading in background
model_thread = threading.Thread(target=load_model_background)
model_thread.start()

# Start GA worker only after model is loaded
def start_ga_worker():
    while model is None or tokenizer is None:
        time.sleep(1)
    worker = GAWorker()
    worker.start()
    print("Background GA worker started.")

ga_thread = threading.Thread(target=start_ga_worker)
ga_thread.start()

###############################
# RUN FLASK
###############################
if __name__ == "__main__":
    # Use port 80 for deployment
    print("Starting Flask app on port 8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)