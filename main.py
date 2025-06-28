####################################
# main.py
# Replit: Continual Mining + Leaderboard
# FIXED: Convert ObservedList to plain list to avoid concat errors
# UPDATED: Use correct Library of Babel page ID format
####################################

import requests
import random
import torch
import time
import threading
import os
import re
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
REQUEST_TIMEOUT = 10       # seconds
MAX_TEXT_LENGTH = 1000
MAX_EMBED_TOKENS = 256

# Library of Babel configuration
HEXAGON_LENGTH = 50  # Use shorter hexagon names for efficiency
WALL_COUNT = 4
SHELF_COUNT = 5
VOLUME_COUNT = 32
PAGES_PER_BOOK = 410

# After each GA "run," how long to sleep (seconds) before next run
SLEEP_BETWEEN_RUNS = 5

# Replit DB KEY where we store top hits
TOP_HITS_DB_KEY = "top_hits"
MAX_STORED_HITS = 20       # keep top 20 in the DB

###############################
# LIBRARY OF BABEL FUNCTIONS
###############################

def generate_hexagon_name(length=None):
    """
    Generate a valid hexagon name for the Library of Babel.
    Uses base-36 (letters and numbers).
    """
    if length is None:
        length = HEXAGON_LENGTH

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

def mutate(page_id, mutation_rate=MUTATION_RATE):
    """
    Mutate a Library of Babel page ID.
    """
    components = parse_page_id(page_id)
    if not components:
        return generate_page_id()  # If invalid, generate new one

    # Mutate hexagon characters
    if random.random() < 0.7:  # 70% chance to mutate hexagon
        hexagon = list(components['hexagon'])
        for i in range(len(hexagon)):
            if random.random() < mutation_rate:
                chars = '0123456789abcdefghijklmnopqrstuvwxyz'
                hexagon[i] = random.choice(chars)
        components['hexagon'] = ''.join(hexagon)

    # Small chance to change location
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

# Backwards compatibility functions
def random_page_id(length=None):
    """For backwards compatibility - now generates proper Library of Babel IDs"""
    return generate_page_id()

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
# PAGE SCRAPING
###############################
def get_page_text(page_id):
    """
    Retrieves the text from a Library of Babel page.
    Now handles the proper page ID format.
    """
    # Validate page ID format
    components = parse_page_id(page_id)
    if not components:
        print(f"[PAGE ERROR] Invalid page ID format: {page_id}")
        return ""

    url = f"https://libraryofbabel.info/book.cgi?{page_id}"

    try:
        print(f"[PAGE FETCH] Attempting to fetch: {url}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        print(f"[PAGE FETCH] Response status: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try multiple selectors to find page content
            selectors = [
                ('div', {'class': 'page'}),
                ('div', {'id': 'page'}),
                ('div', {'class': 'bookcontent'}),
                ('pre', {}),  # Sometimes text is in pre tag
            ]

            for tag, attrs in selectors:
                content_div = soup.find(tag, attrs)
                if content_div:
                    text = content_div.get_text(strip=True)
                    if text and len(text) > 10:  # Make sure we got actual content
                        print(f"[PAGE FETCH] Found content using {tag} {attrs}, length: {len(text)}")
                        return text[:MAX_TEXT_LENGTH]

            # Debug: show what divs are present
            all_divs = soup.find_all('div')
            div_ids = [d.get('id') for d in all_divs if d.get('id')]
            print(f"[PAGE ERROR] No 'page' div found for {page_id[:8]}...")
            print(f"[PAGE ERROR] Found divs with IDs: {div_ids}")

            # If no specific content div found, check if it's an error page
            if "404" in response.text or "not found" in response.text.lower():
                print(f"[PAGE ERROR] Page not found: {page_id}")
            elif "500" in response.text or "error" in response.text.lower():
                print(f"[PAGE ERROR] Server error for page: {page_id}")

            return ""
        else:
            print(f"[PAGE ERROR] HTTP {response.status_code} for page {page_id[:8]}...")
            if response.status_code == 500:
                print(f"[PAGE ERROR] Response content preview: {response.text[:200]}")
            return ""

    except requests.exceptions.Timeout:
        print(f"[PAGE ERROR] Timeout for page {page_id[:8]}...")
        return ""
    except Exception as e:
        print(f"[PAGE ERROR] Exception: {e}")
        return ""

###############################
# SEMANTIC SCORING
###############################
def get_semantic_score_and_embedding(text):
    """
    Returns (score, embedding) tuple.
    """
    if not text or not model or not tokenizer:
        return 0.0, None

    text = text[:MAX_TEXT_LENGTH]

    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', padding=True, 
                          truncation=True, max_length=MAX_EMBED_TOKENS)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

        # Mean pooling
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Calculate coherence score
        embedding_norm = torch.norm(mean_pooled, dim=1).item()

        # Additional text quality metrics
        words = text.split()
        if len(words) < 3:
            return 0.0, mean_pooled

        # Check for repetitive patterns
        unique_words = len(set(words))
        repetition_score = unique_words / len(words) if words else 0

        # Check for actual words vs gibberish
        alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
        word_quality = len(alpha_words) / len(words) if words else 0

        # Combine metrics
        final_score = (embedding_norm * 0.3 + 
                      repetition_score * 0.3 + 
                      word_quality * 0.4) / 10

        return float(final_score), mean_pooled

    except Exception as e:
        print(f"[SCORE ERROR] Error in scoring: {e}")
        return 0.0, None

def interpret_text(snippet, embedding_norm):
    """
    Provide interpretability metrics.
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
    if count == 0:
        return {
            "Token count": 0,
            "Avg word length": 0.0,
            "Alphabetic ratio": 0.0,
            "Repeated word ratio": 0.0,
            "Embedding norm": embedding_norm
        }

    cleaned = [t.strip(",.!?;:\"'()[]{}") for t in tokens]
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

###############################
# GENETIC ALGORITHM
###############################
def evaluate_population(pop):
    results = []
    for page_id in pop:
        full_text = get_page_text(page_id)
        score, _ = get_semantic_score_and_embedding(full_text)
        snippet = full_text[:300].replace("\n", " ")
        metrics = interpret_text(snippet, score)
        results.append({
            "page_id": page_id,
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
    """
    current_hits = load_top_hits()
    updated = False

    for cand in candidates:
        # Skip very low scores
        if cand["score"] < 0.01:
            continue

        # Add timestamp if not present
        if "timestamp" not in cand:
            cand["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # If DB not full, just add
        if len(current_hits) < MAX_STORED_HITS:
            current_hits.append(cand)
            updated = True
        else:
            # Check if this candidate beats the worst
            current_hits_sorted = sorted(current_hits, key=lambda x: x["score"], reverse=True)
            worst_score = current_hits_sorted[-1]["score"]
            if cand["score"] > worst_score:
                # Remove worst
                current_hits.remove(current_hits_sorted[-1])
                # Add new
                current_hits.append(cand)
                updated = True

    if updated:
        # Sort by score desc
        current_hits = sorted(current_hits, key=lambda x: x["score"], reverse=True)
        # Keep only top MAX_STORED_HITS
        current_hits = current_hits[:MAX_STORED_HITS]
        save_top_hits(current_hits)
        print(f"[LEADERBOARD] Updated with new entries. Top score: {current_hits[0]['score']:.4f}")

###############################
# FLASK APP SETUP
###############################
app = Flask(__name__)

# Add a health check endpoint
@app.route("/health")
def health():
    return "OK", 200

@app.route("/test-page")
def test_page():
    """Test endpoint to validate page fetching with a known Library of Babel page"""
    from flask import request

    # Use a known page ID that should work
    test_page_id = request.args.get('page_id', 'test123')

    html = f"""
    <html>
    <head><title>Test Page Fetching</title>
    <style>body{{font-family:monospace;background:#000;color:#0f0;padding:20px;}}</style>
    </head>
    <body>
    <h1>Test Page Fetching</h1>
    <p>Testing page fetch for ID: {test_page_id}</p>
    """

    if test_page_id == 'test123':
        # Return test content
        test_text = "This is a test page with meaningful content to validate the scoring system."
        score, embedding = get_semantic_score_and_embedding(test_text)
        html += f"""
        <h3>Test Mode (not fetching from Library of Babel)</h3>
        <p><strong>Test Text:</strong> {test_text}</p>
        <p><strong>Score:</strong> {score}</p>
        <p><strong>Embedding Available:</strong> {embedding is not None}</p>
        """
    else:
        # Try to fetch from Library of Babel
        page_text = get_page_text(test_page_id)
        if page_text:
            score, embedding = get_semantic_score_and_embedding(page_text)
            html += f"""
            <h3>Retrieved from Library of Babel</h3>
            <p><strong>Text Length:</strong> {len(page_text)} characters</p>
            <p><strong>First 200 chars:</strong> {page_text[:200]}...</p>
            <p><strong>Score:</strong> {score}</p>
            <p><strong>Embedding Available:</strong> {embedding is not None}</p>
            """
        else:
            html += f"""
            <h3>Failed to Retrieve Page</h3>
            <p>Could not get content for page ID: {test_page_id}</p>
            """

    html += """
    <p><a href="/test-page?page_id=test123">Test with sample content</a></p>
    <p><a href="/test-page?page_id=1">Test with page ID "1"</a></p>
    <p><a href="/">Back to Home</a></p>
    </body>
    </html>
    """

    return html

@app.route("/test-scoring")
def test_scoring():
    """Test the scoring system with known content"""
    from flask import request

    # Test with meaningful text samples
    test_texts = [
        "This is a meaningful sentence with proper grammar and structure.",
        "The quick brown fox jumps over the lazy dog in the garden.",
        "Once upon a time, in a land far away, there lived a wise old wizard.",
        "ajsdkfja skdjf aslkdfj alskdfj alskdfj",  # Gibberish
        "a a a a a a a a a a a a a a a a a a",  # Repetitive
        "",  # Empty
    ]

    # Also test with a real page ID if provided
    real_page_id = request.args.get('page_id')

    results = []

    for test_text in test_texts:
        score, _ = get_semantic_score_and_embedding(test_text)
        metrics = interpret_text(test_text, score)
        results.append({
            "test_id": f"test_{len(results)}",
            "text": test_text,
            "score": score,
            "metrics": metrics,
            "embedding_norm": metrics.get("Embedding norm", 0),
            "word_count": len(test_text.split())
        })

    # Test with real page if provided
    if real_page_id:
        page_text = get_page_text(real_page_id)
        if page_text:
            score, _ = get_semantic_score_and_embedding(page_text)
            metrics = interpret_text(page_text[:100], score)
            results.append({
                "test_id": "real_page",
                "page_id": real_page_id,
                "text": page_text[:100] + "...",
                "score": score,
                "metrics": metrics,
                "embedding_norm": metrics.get("Embedding norm", 0),
                "word_count": len(page_text.split())
            })
        else:
            results.append({
                "test_id": "real_page",
                "page_id": real_page_id,
                "error": "Could not retrieve page text"
            })

    return f"""
    <html>
    <head><title>Scoring System Test</title>
    <style>body{{font-family:monospace;background:#000;color:#0f0;padding:20px;}}
    table{{border-collapse:collapse;width:100%;}}
    th,td{{border:1px solid #0f0;padding:8px;text-align:left;}}
    .high-score{{color:#ff0;}}
    .error{{color:#f00;}}
    </style>
    </head>
    <body>
    <h1>Scoring System Test Results</h1>
    <p>Testing the semantic scoring function with known content:</p>
    <table>
    <tr><th>Test ID</th><th>Text Sample</th><th>Score</th><th>Embedding Norm</th><th>Word Count</th></tr>
    """ + ''.join([
        f"<tr><td>{r['test_id']}</td><td>{r['text'][:60]}...</td><td class=\"{('high-score' if r.get('score', 0) > 0.1 else '')}\">{r.get('score', 'ERROR')}</td><td>{r.get('embedding_norm', 'ERROR')}</td><td>{r.get('word_count', 'ERROR')}</td></tr>"
        for r in results
    ]) + """
    </table>
    <p><a href="/test-scoring?page_id=test123">Test with sample page ID</a> | <a href="/">Back to Home</a></p>
    <h3>Expected Results:</h3>
    <ul>
    <li>Meaningful sentences should have scores > 0.1</li>
    <li>Random gibberish should have lower scores</li>
    <li>Empty text should have score 0.0</li>
    </ul>
    </body>
    </html>
    """

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Library of Babel - Evolutionary Semantic Miner</title>
  <style>
    body {
      background: #000;
      color: #0f0;
      font-family: monospace;
      margin: 0;
      padding: 20px;
    }
    .status-panel {
      border: 2px solid #0f0;
      padding: 20px;
      margin: 20px 0;
      background: #001100;
    }
    .evolution-graph {
      height: 200px;
      border: 1px solid #0f0;
      background: #000;
      position: relative;
      margin: 10px 0;
      overflow: hidden;
    }
    .progress-bar {
      height: 20px;
      background: #001100;
      border: 1px solid #0f0;
      margin: 10px 0;
      position: relative;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #0f0, #0ff);
      transition: width 0.3s ease;
    }
    .metric {
      display: inline-block;
      margin: 5px 15px 5px 0;
      padding: 5px 10px;
      border: 1px solid #0f0;
      background: #001100;
    }
    .blinking {
      animation: blink 1s infinite;
    }
    @keyframes blink {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0.3; }
    }
    .evolution-point {
      position: absolute;
      width: 3px;
      background: #0ff;
      bottom: 0;
      transition: height 0.5s ease;
    }
    .current-point {
      background: #ff0 !important;
      box-shadow: 0 0 10px #ff0;
    }
    a { color: #0ff; }
  </style>
</head>
<body>
  <h1>Library of Babel - Evolutionary Semantic Miner</h1>
  <p>This application explores the Library of Babel using genetic algorithms and machine learning.
  The Library of Babel is a Borges inspried website (libraryofbabel.info) that generates pseudo-random text pages based on unique page IDs. Project Tommyknockers attempts to mine the Library of Babel for meaningful text within this vast space of randomness by:
  <ul>
    <li>Generating a random population of page IDs.</li>
    <li>Retrieving and scoring each page's coherence using a language model.</li>
    <li>Keeping the best-performing pages, mutating them, and repeating.</li>
  </ul>
 This is essentially an AI-powered exploration tool trying to find needles (coherent text) in an astronomically large haystack (the Library of Babel's random text space). The prize? Every book, poem or idea that ever has been or ever will be written lies within the Library of Babel...waiting for a clever librarian.</p>

  <div class="status-panel">
    <h2>🧬 Evolution Status</h2>
    <div id="status-indicator" class="blinking">● Initializing...</div>

    <div style="margin: 15px 0;">
      <div class="metric">Run: <span id="current-run">0</span></div>
      <div class="metric">Generation: <span id="current-generation">0</span> / {{ NUM_GENERATIONS_PER_RUN }}</div>
      <div class="metric">Pages Evaluated: <span id="pages-evaluated">0</span></div>
      <div class="metric">Best Score: <span id="best-score">0.000</span></div>
    </div>

    <div>
      <strong>Generation Progress:</strong>
      <div class="progress-bar">
        <div id="generation-progress" class="progress-fill" style="width: 0%;"></div>
      </div>
    </div>

    <div>
      <strong>Evolution History (Last 50 Generations):</strong>
      <div id="evolution-graph" class="evolution-graph"></div>
      <div style="font-size: 0.8em; color: #0ff;">Vertical axis: Semantic Score | Horizontal axis: Generation Progress</div>
    </div>

    <div id="current-best" style="margin-top: 15px; font-size: 0.9em;">
      <strong>Current Best Page:</strong> <span id="best-page-id">None</span>
    </div>
  </div>

  <p>Top discoveries appear in a running scoreboard. Visit the <a href="/leaderboard">leaderboard</a> to see the current best hits discovered so far, or check the <a href="/diagnostics">diagnostics page</a> for detailed system analysis.</p>

  <script>
    function updateStatus() {
      fetch('/api/status')
        .then(response => response.json())
        .then(data => {
          // Update status indicator
          const statusEl = document.getElementById('status-indicator');
          statusEl.textContent = '● ' + data.status;
          statusEl.className = data.status.includes('Running') ? 'blinking' : '';

          // Update metrics
          document.getElementById('current-run').textContent = data.current_run || 0;
          document.getElementById('current-generation').textContent = data.current_generation || 0;
          document.getElementById('pages-evaluated').textContent = data.pages_evaluated || 0;
          document.getElementById('best-score').textContent = (data.best_score_this_run || 0).toFixed(4);

          // Update progress bar
          const progress = ((data.current_generation || 0) / {{ NUM_GENERATIONS_PER_RUN }}) * 100;
          document.getElementById('generation-progress').style.width = progress + '%';

          // Update best page
          if (data.best_page_id) {
            document.getElementById('best-page-id').innerHTML = 
              `<a href="https://libraryofbabel.info/book.cgi?${data.best_page_id}" target="_blank">${data.best_page_id}</a>`;
          }

          // Update evolution graph
          if (data.evolution_history && data.evolution_history.length > 0) {
            updateEvolutionGraph(data.evolution_history);
          }
        })
        .catch(err => console.error('Status update error:', err));
    }

    function updateEvolutionGraph(history) {
      const graph = document.getElementById('evolution-graph');
      graph.innerHTML = ''; // Clear existing

      // Take last 50 points
      const points = history.slice(-50);
      if (points.length === 0) return;

      // Find max score for scaling
      const maxScore = Math.max(...points.map(p => p.score || 0));
      const graphWidth = graph.offsetWidth;
      const graphHeight = graph.offsetHeight;

      points.forEach((point, index) => {
        const x = (index / (points.length - 1)) * graphWidth;
        const y = (point.score / maxScore) * graphHeight * 0.9; // 90% of height

        const bar = document.createElement('div');
        bar.className = 'evolution-point';
        if (index === points.length - 1) {
          bar.className += ' current-point';
        }
        bar.style.left = x + 'px';
        bar.style.height = y + 'px';
        bar.style.width = Math.max(3, graphWidth / points.length - 1) + 'px';

        graph.appendChild(bar);
      });
    }

    // Update every second
    setInterval(updateStatus, 1000);
    updateStatus();
  </script>
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
      margin: 0;
      padding: 20px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
    }
    th, td {
      border: 1px solid #0f0;
      padding: 10px;
      text-align: left;
    }
    th {
      background: #001100;
      color: #0ff;
    }
    tr:hover {
      background: #001100;
    }
    a {
      color: #0ff;
    }
    .snippet {
      font-size: 0.9em;
      color: #ccc;
      max-width: 400px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metrics {
      font-size: 0.8em;
      color: #999;
    }
  </style>
</head>
<body>
  <h1>🏆 Leaderboard - Top Discoveries</h1>
  <p>The highest-scoring pages discovered in the Library of Babel:</p>

  <table>
    <tr>
      <th>Rank</th>
      <th>Page ID</th>
      <th>Score</th>
      <th>Text Preview</th>
      <th>Metrics</th>
      <th>Discovered</th>
    </tr>
    {% for item in leaderboard %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><a href="https://libraryofbabel.info/book.cgi?{{ item.page_id }}" target="_blank">{{ item.page_id[:8] }}...</a></td>
      <td>{{ item.score | round(2) }}</td>
      <td>{{ item.snippet }}</td>
      <td class="metrics">
        <pre>{{ item.metrics }}</pre>
      </td>
      <td>{{ item.timestamp }}</td>
    </tr>
    {% endfor %}
  </table>
  <p style="text-align:center;margin-top:20px;"><a href="/">Back Home</a> | <a href="/diagnostics">Diagnostics</a></p>
</body>
</html>
"""

DIAGNOSTICS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Library of Babel - Diagnostics</title>
  <style>
    body {
      background: #000;
      color: #0f0;
      font-family: monospace;
      margin: 0;
      padding: 20px;
      font-size: 12px;
    }
    .diagnostic-section {
      border: 1px solid #0f0;
      margin: 15px 0;
      padding: 15px;
      background: #001100;
    }
    .diagnostic-section h3 {
      color: #0ff;
      margin-top: 0;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0;
    }
    th, td {
      border: 1px solid #0f0;
      padding: 6px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #002200;
      color: #0ff;
    }
    a {
      color: #0ff;
    }
    .code-block {
      background: #000;
      border: 1px solid #333;
      padding: 10px;
      margin: 5px 0;
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 200px;
      overflow-y: auto;
    }
    .status-good { color: #0f0; }
    .status-warning { color: #ff0; }
    .status-error { color: #f00; }
    .refresh-btn {
      background: #002200;
      color: #0f0;
      border: 1px solid #0f0;
      padding: 5px 10px;
      cursor: pointer;
      margin: 5px;
    }
    .refresh-btn:hover {
      background: #004400;
    }
  </style>
</head>
<body>
  <h1>🔬 System Diagnostics</h1>
  <button class="refresh-btn" onclick="location.reload()">Refresh Data</button>

  <div class="diagnostic-section">
    <h3>📊 Current System Status</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th><th>Status</th></tr>
      <tr><td>Model Loaded</td><td>{{ model_status }}</td><td class="{{ model_status_class }}">{{ model_status_text }}</td></tr>
      <tr><td>GA Worker Running</td><td>{{ ga_status }}</td><td class="{{ ga_status_class }}">{{ ga_status_text }}</td></tr>
      <tr><td>Current Run</td><td>{{ current_run }}</td><td>-</td></tr>
      <tr><td>Current Generation</td><td>{{ current_generation }}</td><td>-</td></tr>
      <tr><td>Total Pages Evaluated</td><td>{{ pages_evaluated }}</td><td>-</td></tr>
      <tr><td>Best Score This Run</td><td>{{ best_score }}</td><td>-</td></tr>
    </table>
  </div>

  <div class="diagnostic-section">
    <h3>🔍 Latest Generation Details</h3>
    <p>Results from the most recent generation evaluated:</p>
    <table>
      <tr><th>Gen#</th><th>Run#</th><th>Time</th><th>Best Score</th><th>Best Page</th><th>Text Sample</th><th>Population</th></tr>
      {% for gen in generation_details[-10:] %}
      <tr>
        <td>{{ gen.generation }}</td>
        <td>{{ gen.run }}</td>
        <td>{{ gen.timestamp }}</td>
        <td class="{{ 'status-good' if gen.best_score > 0.1 else 'status-warning' }}">{{ gen.best_score | round(4) }}</td>
        <td><a href="https://libraryofbabel.info/book.cgi?{{ gen.best_page_id }}" target="_blank">View</a></td>
        <td style="max-width: 200px; overflow: hidden;">{{ gen.text_sample }}</td>
        <td>
          <details>
            <summary>{{ gen.population_size }} pages</summary>
            <div class="code-block">{{ gen.population_details }}</div>
          </details>
        </td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <div class="diagnostic-section">
    <h3>🔍 API Response Test</h3>
    <p>Raw API response from /api/status:</p>
    <div id="api-response" class="code-block">Loading...</div>
    <button class="refresh-btn" onclick="testApiResponse()">Test API</button>
  </div>

  <div class="diagnostic-section">
    <h3>📈 Evolution History Analysis</h3>
    <p>Statistical analysis of {{ evolution_history|length }} recorded generations:</p>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total Generations</td><td>{{ stats.total_generations }}</td></tr>
      <tr><td>Average Score</td><td>{{ stats.avg_score | round(4) }}</td></tr>
      <tr><td>Best Score Ever</td><td>{{ stats.max_score | round(4) }}</td></tr>
      <tr><td>Worst Score</td><td>{{ stats.min_score | round(4) }}</td></tr>
      <tr><td>Score Standard Deviation</td><td>{{ stats.std_score | round(4) }}</td></tr>
      <tr><td>Recent Trend (last 10)</td><td class="{{ stats.trend_class }}">{{ stats.trend_text }}</td></tr>
    </table>
  </div>

  <p style="text-align:center;margin-top:20px;">
    <a href="/">Home</a> | <a href="/leaderboard">Leaderboard</a>
  </p>

  <script>
    function testApiResponse() {
      fetch('/api/status')
        .then(response => response.text())
        .then(text => {
          document.getElementById('api-response').textContent = text;
        })
        .catch(err => {
          document.getElementById('api-response').textContent = 'Error: ' + err.message;
        });
    }

    // Test API on page load
    testApiResponse();
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_HTML, NUM_GENERATIONS_PER_RUN=NUM_GENERATIONS_PER_RUN)

# Global status tracking with persistence
STATUS_DB_KEY = "status_data"
EVOLUTION_HISTORY_DB_KEY = "evolution_history"
GENERATION_DETAILS_DB_KEY = "generation_details"

def load_status_data():
    """Load status data from DB or return defaults"""
    return db.get(STATUS_DB_KEY, {
        "status": "Initializing...",
        "current_run": 0,
        "current_generation": 0,
        "pages_evaluated": 0,
        "best_score_this_run": 0.0,
        "best_page_id": None
    })

def save_status_data(data):
    """Save status data to DB"""
    db[STATUS_DB_KEY] = data

def load_evolution_history():
    """Load evolution history from DB"""
    return list(db.get(EVOLUTION_HISTORY_DB_KEY, []))

def save_evolution_history(history):
    """Save evolution history to DB"""
    db[EVOLUTION_HISTORY_DB_KEY] = history

def load_generation_details():
    """Load detailed generation results from DB"""
    return list(db.get(GENERATION_DETAILS_DB_KEY, []))

def save_generation_details(details):
    """Save detailed generation results to DB"""
    db[GENERATION_DETAILS_DB_KEY] = details

# Initialize from DB
status_data = load_status_data()
evolution_history = load_evolution_history()
generation_details = load_generation_details()

@app.route("/api/status")
def api_status():
    from flask import jsonify
    import math
    import json

    try:
        # Ensure all data is JSON serializable and handle NaN/Inf values
        best_score = status_data.get("best_score_this_run", 0.0)
        if not isinstance(best_score, (int, float)) or math.isnan(best_score) or math.isinf(best_score):
            best_score = 0.0

        # Clean evolution history
        clean_history = []
        for item in evolution_history[-50:]:  # Last 50 entries
            score = item.get("score", 0.0)
            if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
                score = 0.0
            clean_history.append({
                "generation": item.get("generation", 0),
                "run": item.get("run", 0),
                "score": float(score)
            })

        response_data = {
            "status": status_data.get("status", "Unknown"),
            "current_run": int(status_data.get("current_run", 0)),
            "current_generation": int(status_data.get("current_generation", 0)),
            "pages_evaluated": int(status_data.get("pages_evaluated", 0)),
            "best_score_this_run": float(best_score),
            "best_page_id": status_data.get("best_page_id"),
            "evolution_history": clean_history
        }

        # Ensure JSON serialization works
        json.dumps(response_data)  # Test serialization

        return jsonify(response_data)
    except Exception as e:
        print(f"[API ERROR] Error in /api/status: {e}")
        return jsonify({
            "status": "Error",
            "error": str(e),
            "current_run": 0,
            "current_generation": 0,
            "pages_evaluated": 0,
            "best_score_this_run": 0.0,
            "best_page_id": None,
            "evolution_history": []
        })

@app.route("/leaderboard")
def leaderboard():
    hits = load_top_hits()
    return render_template_string(LEADERBOARD_TEMPLATE, leaderboard=hits)

@app.route("/diagnostics")
def diagnostics():
    # Calculate stats
    scores = [h.get("score", 0) for h in evolution_history if isinstance(h.get("score", 0), (int, float))]

    if scores:
        import numpy as np
        stats = {
            "total_generations": len(evolution_history),
            "avg_score": np.mean(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "std_score": np.std(scores),
            "trend_class": "status-good" if len(scores) > 10 and np.mean(scores[-10:]) > np.mean(scores[:10]) else "status-warning",
            "trend_text": "Improving" if len(scores) > 10 and np.mean(scores[-10:]) > np.mean(scores[:10]) else "Stable"
        }
    else:
        stats = {
            "total_generations": 0,
            "avg_score": 0,
            "max_score": 0,
            "min_score": 0,
            "std_score": 0,
            "trend_class": "status-warning",
            "trend_text": "No data"
        }

    return render_template_string(
        DIAGNOSTICS_TEMPLATE,
        model_status=model is not None,
        model_status_class="status-good" if model is not None else "status-error",
        model_status_text="Loaded" if model is not None else "Not loaded",
        ga_status=ga_worker_running,
        ga_status_class="status-good" if ga_worker_running else "status-warning",
        ga_status_text="Running" if ga_worker_running else "Idle",
        current_run=status_data.get("current_run", 0),
        current_generation=status_data.get("current_generation", 0),
        pages_evaluated=status_data.get("pages_evaluated", 0),
        best_score=status_data.get("best_score_this_run", 0),
        generation_details=generation_details,
        evolution_history=evolution_history,
        stats=stats
    )

###############################
# CONTINUAL GA WORKER
###############################
ga_worker_running = False

def ga_worker():
    """
    Background worker that continuously runs the GA.
    """
    global ga_worker_running, status_data, evolution_history, generation_details

    print("Background GA worker started.")
    ga_worker_running = True

    run_counter = status_data.get("current_run", 0)

    while True:
        try:
            run_counter += 1
            status_data["current_run"] = run_counter
            status_data["status"] = f"Running evolution (Run #{run_counter})..."
            save_status_data(status_data)

            # 1. Generate initial population
            print(f"[GA DEBUG] Starting generation 1/{NUM_GENERATIONS_PER_RUN} of run {run_counter}")
            population = [random_page_id() for _ in range(POPULATION_SIZE)]

            run_best_score = 0.0
            run_best_page = None

            # 2. Run evolution for NUM_GENERATIONS_PER_RUN
            for gen in range(NUM_GENERATIONS_PER_RUN):
                status_data["current_generation"] = gen + 1
                status_data["status"] = f"Run {run_counter}, Generation {gen + 1}/{NUM_GENERATIONS_PER_RUN}"
                save_status_data(status_data)

                print(f"[GA DEBUG] Evaluating population of {len(population)} pages...")
                evaluated = evaluate_population(population)

                # Update pages evaluated count
                status_data["pages_evaluated"] = status_data.get("pages_evaluated", 0) + len(evaluated)

                # Track best in this generation
                if evaluated:
                    gen_best = max(evaluated, key=lambda x: x["score"])
                    print(f"[GA DEBUG] Generation {gen + 1} best score: {gen_best['score']:.4f}")

                    # Update evolution history
                    evolution_history.append({
                        "generation": gen + 1,
                        "run": run_counter,
                        "score": gen_best["score"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Keep only last 100 entries to prevent DB bloat
                    if len(evolution_history) > 100:
                        evolution_history = evolution_history[-100:]
                    save_evolution_history(evolution_history)

                    # Save generation details
                    generation_details.append({
                        "generation": gen + 1,
                        "run": run_counter,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "best_score": gen_best["score"],
                        "best_page_id": gen_best["page_id"],
                        "text_sample": gen_best["snippet"][:100] + "...",
                        "population_size": len(population),
                        "population_details": "\n".join([f"{e['page_id'][:8]}... : {e['score']:.4f}" for e in evaluated[:5]])
                    })

                    # Keep only last 50 generation details
                    if len(generation_details) > 50:
                        generation_details = generation_details[-50:]
                    save_generation_details(generation_details)

                    # Track best in run
                    if gen_best["score"] > run_best_score:
                        run_best_score = gen_best["score"]
                        run_best_page = gen_best
                        status_data["best_score_this_run"] = run_best_score
                        status_data["best_page_id"] = gen_best["page_id"]
                        save_status_data(status_data)
                else:
                    print("[GA DEBUG] Warning: No evaluated results!")

                # Update leaderboard
                try_update_leaderboard(evaluated)

                # Breed next generation (unless last generation)
                if gen < NUM_GENERATIONS_PER_RUN - 1:
                    parents = select_parents(evaluated, KEEP_RATIO)
                    population = breed_population(parents, POPULATION_SIZE)

                    # Add some fresh blood
                    for _ in range(max(1, POPULATION_SIZE // 5)):
                        population[random.randint(0, len(population) - 1)] = random_page_id()

                    # Apply mutation
                    population = [mutate(p) if random.random() < 0.3 else p for p in population]

            # 3. Run complete
            print(f"[GA DEBUG] Run {run_counter} complete. Best score: {run_best_score:.4f}")
            if run_best_page:
                print(f"[GA DEBUG] Best page: {run_best_page['page_id']}")

            # 4. Sleep before next run
            status_data["status"] = f"Sleeping for {SLEEP_BETWEEN_RUNS}s before next run..."
            save_status_data(status_data)
            time.sleep(SLEEP_BETWEEN_RUNS)

        except Exception as e:
            print(f"[GA ERROR] Error in GA worker: {e}")
            import traceback
            traceback.print_exc()
            status_data["status"] = f"Error: {str(e)}"
            save_status_data(status_data)
            time.sleep(10)  # Sleep longer on error

def start_flask_and_ga():
    """
    Start Flask in main thread and GA worker in background.
    """
    # Load model in background thread
    model_thread = threading.Thread(target=load_model_background)
    model_thread.daemon = True
    model_thread.start()

    # Start GA worker in background thread
    ga_thread = threading.Thread(target=ga_worker)
    ga_thread.daemon = True
    ga_thread.start()

    # Start Flask in main thread
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    start_flask_and_ga()