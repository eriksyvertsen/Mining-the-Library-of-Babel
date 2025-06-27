import requests
from bs4 import BeautifulSoup
import random
import re
import hashlib
from flask import Flask, jsonify, request, render_template_string
from datetime import datetime
import time
import threading
from replit import db
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

###############################
# CONFIGURATIONS
###############################
POPULATION_SIZE = 10
NUM_GENERATIONS_PER_RUN = 5
NUM_RUNS = 10
KEEP_RATIO = 0.5
MUTATION_RATE = 0.02
TOP_HITS_DB_KEY = "top_hits"
MAX_LEADERBOARD_SIZE = 50

# Library of Babel configuration
HEXAGON_LENGTH = 3200  # Maximum length of hexagon names according to documentation
WALL_COUNT = 4
SHELF_COUNT = 5
VOLUME_COUNT = 32
PAGES_PER_BOOK = 410

###############################
# LIBRARY OF BABEL FUNCTIONS
###############################

def generate_hexagon_name(length=None):
    """
    Generate a valid hexagon name for the Library of Babel.
    Hexagon names can be extremely long (up to 3200 characters) but we'll use shorter ones for efficiency.
    """
    if length is None:
        # Use various lengths, but not too long for URL compatibility
        length = random.choice([10, 20, 50, 100, 200])

    # Library of Babel uses base-36 (letters and numbers)
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(chars) for _ in range(length))

def generate_page_id():
    """
    Generate a complete Library of Babel page identifier.
    Format: hexagon-wWALL-sSHELF-vVOLUME:PAGE
    """
    hexagon = generate_hexagon_name()
    wall = random.randint(1, WALL_COUNT)
    shelf = random.randint(1, SHELF_COUNT)
    volume = random.randint(1, VOLUME_COUNT)
    page = random.randint(1, PAGES_PER_BOOK)

    return f"{hexagon}-w{wall}-s{shelf}-v{volume}:{page}"

def parse_page_id(page_id):
    """
    Parse a page ID into its components.
    Returns dict with keys: hexagon, wall, shelf, volume, page
    """
    # Match pattern: hexagon-wN-sN-vN:N
    pattern = r'^([^-]+)-w(\d+)-s(\d+)-v(\d+):(\d+)$'
    match = re.match(pattern, page_id)

    if match:
        return {
            'hexagon': match.group(1),
            'wall': int(match.group(2)),
            'shelf': int(match.group(3)),
            'volume': int(match.group(4)),
            'page': int(match.group(5))
        }
    return None

def get_page_text(page_id):
    """
    Retrieves the text from a Library of Babel page given its page ID.
    """
    # Parse the page ID to ensure it's valid
    components = parse_page_id(page_id)
    if not components:
        print(f"[SCRAPER] Invalid page ID format: {page_id}")
        return ""

    # The Library of Babel URL format
    url = f"https://libraryofbabel.info/book.cgi?{page_id}"

    try:
        print(f"[SCRAPER] Fetching: {url}")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try multiple selectors to find the page content
            selectors = [
                ('div', {'class': 'page'}),
                ('div', {'id': 'page'}),
                ('div', {'class': 'bookcontent'}),
                ('pre', {}),  # Sometimes the text is in a pre tag
            ]

            for tag, attrs in selectors:
                content_element = soup.find(tag, attrs)
                if content_element:
                    text = content_element.get_text()
                    if text and len(text) > 10:  # Make sure we got actual content
                        print(f"[SCRAPER] Found content using selector {tag} {attrs}, length: {len(text)}")
                        return text

            # If no specific content div found, look for any substantial text
            # This is a fallback - the page might have a different structure
            all_text = soup.get_text()
            if len(all_text) > 100:
                print(f"[SCRAPER] Using fallback text extraction, length: {len(all_text)}")
                return all_text

        print(f"[SCRAPER] Failed to retrieve content, status: {response.status_code}")
        return ""

    except Exception as e:
        print(f"[SCRAPER] Error fetching {url}: {e}")
        return ""

def mutate(page_id, mutation_rate):
    """
    Mutate a Library of Babel page ID.
    Can change hexagon characters, or jump to nearby walls/shelves/volumes/pages.
    """
    components = parse_page_id(page_id)
    if not components:
        return generate_page_id()  # If invalid, generate new one

    # Decide what to mutate
    if random.random() < 0.7:  # 70% chance to mutate hexagon
        hexagon = list(components['hexagon'])
        for i in range(len(hexagon)):
            if random.random() < mutation_rate:
                chars = '0123456789abcdefghijklmnopqrstuvwxyz'
                hexagon[i] = random.choice(chars)
        components['hexagon'] = ''.join(hexagon)

    # Small chance to change location within the library
    if random.random() < 0.1:
        components['wall'] = random.randint(1, WALL_COUNT)
    if random.random() < 0.1:
        components['shelf'] = random.randint(1, SHELF_COUNT)
    if random.random() < 0.1:
        components['volume'] = random.randint(1, VOLUME_COUNT)
    if random.random() < 0.2:  # Higher chance to change page
        components['page'] = random.randint(1, PAGES_PER_BOOK)

    return f"{components['hexagon']}-w{components['wall']}-s{components['shelf']}-v{components['volume']}:{components['page']}"

def crossover(page_id1, page_id2):
    """
    Crossover two Library of Babel page IDs.
    """
    comp1 = parse_page_id(page_id1)
    comp2 = parse_page_id(page_id2)

    if not comp1 or not comp2:
        return generate_page_id()

    # Crossover hexagon names
    hex1 = comp1['hexagon']
    hex2 = comp2['hexagon']

    # Make sure both hexagons are similar length
    min_len = min(len(hex1), len(hex2))
    if min_len > 0:
        crossover_point = random.randint(0, min_len - 1)
        new_hexagon = hex1[:crossover_point] + hex2[crossover_point:min_len]
    else:
        new_hexagon = generate_hexagon_name()

    # Randomly pick other components from either parent
    new_components = {
        'hexagon': new_hexagon,
        'wall': random.choice([comp1['wall'], comp2['wall']]),
        'shelf': random.choice([comp1['shelf'], comp2['shelf']]),
        'volume': random.choice([comp1['volume'], comp2['volume']]),
        'page': random.choice([comp1['page'], comp2['page']])
    }

    return f"{new_components['hexagon']}-w{new_components['wall']}-s{new_components['shelf']}-v{new_components['volume']}:{new_components['page']}"

###############################
# SEMANTIC SCORING
###############################

print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()
print("Model loaded.")

def get_semantic_score_and_embedding(text):
    """
    Returns (score, embedding) tuple.
    Score is a float indicating semantic coherence.
    Embedding is the sentence embedding tensor.
    """
    if not text or len(text.strip()) < 10:
        return 0.0, None

    try:
        # Truncate to reasonable length
        text_snippet = text[:500].strip()

        # Tokenize
        inputs = tokenizer(text_snippet, return_tensors='pt', 
                          padding=True, truncation=True, max_length=128)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask

            # Calculate coherence score based on embedding properties
            embedding_norm = torch.norm(sentence_embedding, dim=1).item()

            # Additional text quality metrics
            words = text_snippet.split()
            if len(words) < 3:
                return 0.0, sentence_embedding

            # Check for repetitive patterns (common in Library of Babel)
            unique_words = len(set(words))
            repetition_score = unique_words / len(words) if words else 0

            # Check for actual words vs gibberish
            alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
            word_quality = len(alpha_words) / len(words) if words else 0

            # Combine metrics
            final_score = (embedding_norm * 0.3 + 
                          repetition_score * 0.3 + 
                          word_quality * 0.4) / 10

            return float(final_score), sentence_embedding

    except Exception as e:
        print(f"[SCORE ERROR] Error in scoring: {e}")
        return 0.0, None

###############################
# GENETIC ALGORITHM
###############################

def evaluate_population(population):
    """
    Evaluate a population of page IDs.
    Returns list of dicts with page_id, score, snippet, metrics.
    """
    results = []

    for page_id in population:
        text = get_page_text(page_id)
        score, _ = get_semantic_score_and_embedding(text)

        # Create snippet
        snippet = text[:300].replace("\n", " ") if text else "[No content retrieved]"

        # Calculate metrics
        metrics = {
            "length": len(text),
            "words": len(text.split()) if text else 0,
            "unique_words": len(set(text.split())) if text else 0
        }

        results.append({
            "page_id": page_id,
            "score": score,
            "snippet": snippet,
            "metrics": metrics,
            "timestamp": datetime.now()
        })

        # Add delay to be respectful to the server
        time.sleep(0.5)

    return results

def select_parents(evaluated, keep_ratio):
    """Select the best performing individuals as parents."""
    sorted_eval = sorted(evaluated, key=lambda x: x["score"], reverse=True)
    k = max(1, int(len(sorted_eval) * keep_ratio))
    return sorted_eval[:k]

def breed_population(parents, new_size):
    """Create new population through crossover and mutation."""
    parent_ids = [p["page_id"] for p in parents]
    new_pop = []

    # Keep best parents (elitism)
    for p in parent_ids[:2]:
        new_pop.append(p)

    # Generate children
    while len(new_pop) < new_size:
        if len(parent_ids) >= 2:
            p1 = random.choice(parent_ids)
            p2 = random.choice(parent_ids)
            child = crossover(p1, p2)
        else:
            # If only one parent, mutate it
            child = mutate(parent_ids[0], MUTATION_RATE * 2)

        # Apply mutation
        if random.random() < 0.3:
            child = mutate(child, MUTATION_RATE)

        new_pop.append(child)

    return new_pop[:new_size]

###############################
# FLASK APP & WEB INTERFACE
###############################

app = Flask(__name__)

# Global state
evolution_status = {
    "status": "Idle",
    "current_run": 0,
    "current_generation": 0,
    "pages_evaluated": 0,
    "best_score": 0.0,
    "best_page_id": None,
    "generation_scores": []
}

def run_evolution():
    """Main evolution loop."""
    global evolution_status

    evolution_status["status"] = "Running evolution..."
    evolution_status["generation_scores"] = []

    for run in range(NUM_RUNS):
        evolution_status["current_run"] = run + 1

        # Initialize population
        population = [generate_page_id() for _ in range(POPULATION_SIZE)]

        run_best_score = 0.0
        run_best_page = None

        for gen in range(NUM_GENERATIONS_PER_RUN):
            evolution_status["current_generation"] = gen + 1
            evolution_status["status"] = f"Run {run + 1}/{NUM_RUNS}, Generation {gen + 1}/{NUM_GENERATIONS_PER_RUN}"

            # Evaluate
            evaluated = evaluate_population(population)
            evolution_status["pages_evaluated"] += len(evaluated)

            # Track best
            gen_best = max(evaluated, key=lambda x: x["score"])
            evolution_status["generation_scores"].append(gen_best["score"])

            if gen_best["score"] > run_best_score:
                run_best_score = gen_best["score"]
                run_best_page = gen_best

                if run_best_score > evolution_status["best_score"]:
                    evolution_status["best_score"] = run_best_score
                    evolution_status["best_page_id"] = gen_best["page_id"]

            # Update leaderboard
            try_update_leaderboard(evaluated)

            # Breed next generation
            if gen < NUM_GENERATIONS_PER_RUN - 1:
                parents = select_parents(evaluated, KEEP_RATIO)
                population = breed_population(parents, POPULATION_SIZE)

            print(f"[EVOLUTION] Run {run + 1}, Gen {gen + 1}: Best score = {gen_best['score']:.4f}")

    evolution_status["status"] = "Evolution complete"
    print("[EVOLUTION] All runs complete!")

def try_update_leaderboard(candidates):
    """Update the leaderboard with new high-scoring pages."""
    current_hits = load_top_hits()

    for candidate in candidates:
        if candidate["score"] > 0.01:  # Only consider pages with meaningful scores
            # Check if this page is already in leaderboard
            existing = next((h for h in current_hits if h["page_id"] == candidate["page_id"]), None)

            if existing:
                # Update if score improved
                if candidate["score"] > existing["score"]:
                    existing.update(candidate)
            else:
                # Add new entry
                current_hits.append(candidate)

    # Sort and trim
    current_hits.sort(key=lambda x: x["score"], reverse=True)
    current_hits = current_hits[:MAX_LEADERBOARD_SIZE]

    save_top_hits(current_hits)

def load_top_hits():
    """Load top hits from database."""
    data = db.get(TOP_HITS_DB_KEY, [])
    return list(data)

def save_top_hits(hits_list):
    """Save top hits to database."""
    db[TOP_HITS_DB_KEY] = hits_list

###############################
# WEB ROUTES
###############################

@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/api/status")
def api_status():
    return jsonify(evolution_status)

@app.route("/api/start")
def api_start():
    # Start evolution in background thread
    thread = threading.Thread(target=run_evolution)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "Evolution started"})

@app.route("/leaderboard")
def leaderboard():
    hits = load_top_hits()
    return render_template_string(LEADERBOARD_HTML, hits=hits)

@app.route("/diagnostics")
def diagnostics():
    # Test page generation
    test_results = []

    for i in range(5):
        page_id = generate_page_id()
        components = parse_page_id(page_id)
        text = get_page_text(page_id)

        test_results.append({
            "page_id": page_id,
            "components": components,
            "text_length": len(text) if text else 0,
            "text_preview": text[:100] if text else "[No content]"
        })

        time.sleep(0.5)  # Be respectful

    return render_template_string(DIAGNOSTICS_HTML, 
                                 test_results=test_results,
                                 status=evolution_status)

# HTML Templates
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Library of Babel Explorer</title>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
        .status { border: 1px solid #0f0; padding: 10px; margin: 10px 0; }
        a { color: #0ff; }
        button { background: #0f0; color: #000; border: none; padding: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Library of Babel Explorer</h1>
    <p>Mining the Library of Babel for coherent text using genetic algorithms.</p>

    <div class="status">
        <h2>Status</h2>
        <div id="status">Loading...</div>
        <button onclick="startEvolution()">Start Evolution</button>
    </div>

    <p>
        <a href="/leaderboard">View Leaderboard</a> | 
        <a href="/diagnostics">Diagnostics</a>
    </p>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerHTML = `
                        Status: ${data.status}<br>
                        Run: ${data.current_run}<br>
                        Generation: ${data.current_generation}<br>
                        Pages Evaluated: ${data.pages_evaluated}<br>
                        Best Score: ${data.best_score.toFixed(4)}<br>
                        Best Page: ${data.best_page_id || 'None'}
                    `;
                });
        }

        function startEvolution() {
            fetch('/api/start')
                .then(r => r.json())
                .then(data => alert(data.status));
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""

LEADERBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Leaderboard - Library of Babel Explorer</title>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #0f0; padding: 8px; text-align: left; }
        a { color: #0ff; }
    </style>
</head>
<body>
    <h1>Leaderboard</h1>
    <p><a href="/">Back to Home</a></p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Score</th>
            <th>Page ID</th>
            <th>Preview</th>
            <th>Link</th>
        </tr>
        {% for hit in hits %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ "%.4f"|format(hit.score) }}</td>
            <td>{{ hit.page_id[:30] }}...</td>
            <td>{{ hit.snippet[:100] }}...</td>
            <td><a href="https://libraryofbabel.info/book.cgi?{{ hit.page_id }}" target="_blank">View</a></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

DIAGNOSTICS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Diagnostics - Library of Babel Explorer</title>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
        .test { border: 1px solid #0f0; padding: 10px; margin: 10px 0; }
        a { color: #0ff; }
        pre { background: #001100; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Diagnostics</h1>
    <p><a href="/">Back to Home</a></p>

    <h2>Page Generation Test</h2>
    {% for test in test_results %}
    <div class="test">
        <h3>Test {{ loop.index }}</h3>
        <p><strong>Page ID:</strong> {{ test.page_id }}</p>
        <p><strong>Components:</strong></p>
        <pre>{{ test.components }}</pre>
        <p><strong>Text Length:</strong> {{ test.text_length }} characters</p>
        <p><strong>Preview:</strong> {{ test.text_preview }}</p>
        <p><a href="https://libraryofbabel.info/book.cgi?{{ test.page_id }}" target="_blank">View on Library of Babel</a></p>
    </div>
    {% endfor %}

    <h2>Evolution Status</h2>
    <pre>{{ status }}</pre>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)