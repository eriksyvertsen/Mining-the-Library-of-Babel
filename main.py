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
        "asdfghjkl qwerty random gibberish nonsense",
        "aaaa bbbb cccc dddd eeee ffff",
        ""
    ]

    results = []
    for i, text in enumerate(test_texts):
        score, embedding = get_semantic_score_and_embedding(text)
        norm = torch.norm(embedding).item() if embedding is not None else 0.0

        results.append({
            "test_id": i + 1,
            "text": text,
            "score": round(score, 6),
            "embedding_norm": round(norm, 6) if embedding is not None else "None",
            "text_length": len(text),
            "word_count": len(text.split()) if text else 0
        })

    # Also test a real Library of Babel page if page_id is provided
    page_id = request.args.get('page_id')
    if page_id:
        page_text = get_page_text(page_id)
        if page_text:
            score, embedding = get_semantic_score_and_embedding(page_text)
            norm = torch.norm(embedding).item() if embedding is not None else 0.0
            results.append({
                "test_id": "real_page",
                "page_id": page_id,
                "text": page_text[:200] + "..." if len(page_text) > 200 else page_text,
                "full_text_length": len(page_text),
                "score": round(score, 6),
                "embedding_norm": round(norm, 6) if embedding is not None else "None",
                "word_count": len(page_text.split()) if page_text else 0
            })
        else:
            results.append({
                "test_id": "real_page",
                "page_id": page_id,
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
          document.getElementById('current-run').textContent = data.current_run;
          document.getElementById('current-generation').textContent = data.current_generation;
          document.getElementById('pages-evaluated').textContent = data.pages_evaluated;
          document.getElementById('best-score').textContent = data.best_score_this_run.toFixed(2);

          // Update progress bar
          const progressPercent = (data.current_generation / 2) * 100;
          document.getElementById('generation-progress').style.width = progressPercent + '%';

          // Update best page
          const bestPageText = data.best_page_id ? 
            `<a href="https://libraryofbabel.info/book.cgi?${data.best_page_id}" target="_blank">${data.best_page_id.substring(0, 12)}...</a>` :
            'None';
          document.getElementById('best-page-id').innerHTML = bestPageText;

          // Update evolution graph
          updateEvolutionGraph(data.evolution_history);
        })
        .catch(err => console.log('Status update failed:', err));
    }

    function updateEvolutionGraph(history) {
      const graph = document.getElementById('evolution-graph');
      graph.innerHTML = '';

      if (history.length === 0) return;

      const maxScore = Math.max(...history.map(h => h.score), 0.1); // Minimum of 0.1 for scaling
      const graphWidth = graph.clientWidth;
      const displayHistory = history.slice(-50); // Show last 50 points
      const pointWidth = Math.max(2, graphWidth / Math.max(displayHistory.length, 50));

      displayHistory.forEach((point, index) => {
        const normalizedHeight = (point.score / maxScore) * 160; // 160px max height
        const height = Math.max(normalizedHeight, 2); // Minimum 2px height to show all points
        const left = index * pointWidth;

        const pointEl = document.createElement('div');
        pointEl.className = 'evolution-point';
        if (index === displayHistory.length - 1) pointEl.className += ' current-point';
        pointEl.style.left = left + 'px';
        pointEl.style.height = height + 'px';
        pointEl.style.width = pointWidth + 'px';
        pointEl.title = `Run ${point.run || 'N/A'} Gen ${point.generation}: Score ${point.score.toFixed(2)}`;

        graph.appendChild(pointEl);
      });
    }

    // Update every 2 seconds
    setInterval(updateStatus, 2000);
    updateStatus(); // Initial load
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
    <h3>⚙️ Model Configuration</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Population Size</td><td>{{ population_size }}</td></tr>
      <tr><td>Generations Per Run</td><td>{{ generations_per_run }}</td></tr>
      <tr><td>Keep Ratio</td><td>{{ keep_ratio }}</td></tr>
      <tr><td>Mutation Rate</td><td>{{ mutation_rate }}</td></tr>
      <tr><td>Page ID Length</td><td>{{ page_id_length }}</td></tr>
      <tr><td>Request Timeout</td><td>{{ request_timeout }}s</td></tr>
      <tr><td>Max Text Length</td><td>{{ max_text_length }}</td></tr>
      <tr><td>Max Embedding Tokens</td><td>{{ max_embed_tokens }}</td></tr>
    </table>
  </div>

  <div class="diagnostic-section">
    <h3>📋 Recent Generation Details</h3>
    <p>Last {{ recent_generations|length }} generations with detailed results:</p>
    <table>
      <tr>
        <th>Run</th>
        <th>Gen</th>
        <th>Best Score</th>
        <th>Best Page ID</th>
        <th>Link</th>
        <th>Text Sample</th>
        <th>Population Details</th>
      </tr>
      {% for gen in recent_generations %}
      <tr>
        <td>{{ gen.run }}</td>
        <td>{{ gen.generation }}</td>
        <td>{{ gen.best_score | round(4) }}</td>
        <td style="font-size: 10px;">{{ gen.best_page_id }}</td>
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

        response = {
            "status": str(status_data.get("status", "Unknown")),
            "current_run": int(status_data.get("current_run", 0)),
            "current_generation": int(status_data.get("current_generation", 0)),
            "pages_evaluated": int(status_data.get("pages_evaluated", 0)),
            "best_score_this_run": float(best_score),
            "best_page_id": str(status_data.get("best_page_id", "")) if status_data.get("best_page_id") else None
        }

        # Convert evolution_history to plain Python list/dicts with better error handling
        evolution_history_plain = []
        for item in evolution_history[-50:]:  # Only send last 50 items to reduce payload
            try:
                if not isinstance(item, dict):
                    continue

                score = item.get("score", 0.0)
                if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
                    score = 0.0

                generation = item.get("generation", 0)
                if not isinstance(generation, (int, float)):
                    generation = 0

                run = item.get("run", 0)
                if not isinstance(run, (int, float)):
                    run = 0

                evolution_history_plain.append({
                    "generation": int(generation),
                    "score": float(score),
                    "run": int(run)
                })
            except (TypeError, AttributeError, ValueError) as e:
                print(f"[API ERROR] Skipping malformed evolution history entry: {e}")
                continue

        response["evolution_history"] = evolution_history_plain

        # Test JSON serialization before returning
        try:
            json.dumps(response)
        except (TypeError, ValueError) as e:
            print(f"[API ERROR] JSON serialization failed: {e}")
            # Return minimal safe response
            return jsonify({
                "status": "JSON Error",
                "current_run": 0,
                "current_generation": 0,
                "pages_evaluated": 0,
                "best_score_this_run": 0.0,
                "best_page_id": None,
                "evolution_history": []
            })

        return jsonify(response)

    except Exception as e:
        print(f"[API ERROR] Exception in api_status: {e}")
        # Return a safe fallback response if anything goes wrong
        return jsonify({
            "status": "Error in API",
            "current_run": 0,
            "current_generation": 0,
            "pages_evaluated": 0,
            "best_score_this_run": 0.0,
            "best_page_id": None,
            "evolution_history": [],
            "error": str(e)
        })

@app.route("/leaderboard")
def leaderboard():
    # Get top hits from DB
    hits = db.get(TOP_HITS_DB_KEY, [])
    # Convert ObservedList to normal list
    hits = list(hits)
    # Sort by score descending
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)
    return render_template_string(LEADERBOARD_TEMPLATE, hits=hits_sorted, max_stored=MAX_STORED_HITS)

@app.route("/diagnostics")
def diagnostics():
    import statistics

    # System status
    model_loaded = model is not None and tokenizer is not None
    model_status = "✓ Loaded" if model_loaded else "✗ Not Loaded"
    model_status_class = "status-good" if model_loaded else "status-error"
    model_status_text = "OK" if model_loaded else "ERROR"

    # GA Worker status (simplified check)
    ga_status = "✓ Running" if status_data.get("current_run", 0) > 0 else "? Unknown"
    ga_status_class = "status-good" if status_data.get("current_run", 0) > 0 else "status-warning"
    ga_status_text = "OK" if status_data.get("current_run", 0) > 0 else "UNKNOWN"

    # Recent generation details
    recent_gens = load_generation_details()[-20:]  # Last 20 generations

    # Evolution history statistics
    evo_hist = load_evolution_history()
    stats = {
        "total_generations": len(evo_hist),
        "avg_score": statistics.mean([h["score"] for h in evo_hist]) if evo_hist else 0,
        "max_score": max([h["score"] for h in evo_hist]) if evo_hist else 0,
        "min_score": min([h["score"] for h in evo_hist]) if evo_hist else 0,
        "std_score": statistics.stdev([h["score"] for h in evo_hist]) if len(evo_hist) > 1 else 0
    }

    # Trend analysis
    if len(evo_hist) >= 10:
        recent_scores = [h["score"] for h in evo_hist[-10:]]
        older_scores = [h["score"] for h in evo_hist[-20:-10]] if len(evo_hist) >= 20 else recent_scores
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)

        if recent_avg > older_avg * 1.1:
            stats["trend_text"] = "Improving ↗"
            stats["trend_class"] = "status-good"
        elif recent_avg < older_avg * 0.9:
            stats["trend_text"] = "Declining ↘"
            stats["trend_class"] = "status-warning"
        else:
            stats["trend_text"] = "Stable →"
            stats["trend_class"] = "status-good"
        else:
            stats["trend_text"] = "Not enough data"
            stats["trend_class"] = "status-warning"

    return render_template_string(DIAGNOSTICS_TEMPLATE,
        # System status
        model_status=model_status,
        model_status_class=model_status_class,
        model_status_text=model_status_text,
        ga_status=ga_status,
        ga_status_class=ga_status_class,
        ga_status_text=ga_status_text,
        current_run=```text
status_data.get("current_run", 0),
        current_generation=status_data.get("current_generation", 0),
        pages_evaluated=status_data.get("pages_evaluated", 0),
        best_score=status_data.get("best_score_this_run", 0.0),

        # Configuration
        population_size=POPULATION_SIZE,
        generations_per_run=NUM_GENERATIONS_PER_RUN,
        keep_ratio=KEEP_RATIO,
        mutation_rate=MUTATION_RATE,
        page_id_length=PAGE_ID_LENGTH,
        request_timeout=REQUEST_TIMEOUT,
        max_text_length=MAX_TEXT_LENGTH,
        max_embed_tokens=MAX_EMBED_TOKENS,

        # Data
        recent_generations=recent_gens,
        evolution_history=evo_hist,
        stats=stats
    )

###############################
# GA & SCRAPING LOGIC
###############################
def get_page_text(page_id: str) -> str:
    url = f"https://libraryofbabel.info/book.cgi?{page_id}"
    print(f"[PAGE FETCH] Attempting to fetch: {url} (ID: {page_id[:12]}...)")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        print(f"[PAGE FETCH] Response status: {response.status_code}")

        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try multiple selectors for the page content
            page_div = soup.find("div", {"id": "page"})
            if not page_div:
                page_div = soup.find("div", {"id": "real"})  # Alternative selector
            if not page_div:
                # Try to find any div with text content
                content_divs = soup.find_all("div")
                for div in content_divs:
                    text = div.get_text().strip()
                    if len(text) > 100:  # Arbitrary threshold for meaningful content
                        page_div = div
                        break

            if page_div:
                text = page_div.get_text().strip()
                # Calculate a simple hash to detect if we're getting the same content
                import hashlib
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                print(f"[PAGE SUCCESS] Retrieved {len(text)} characters for page {page_id[:8]}... (hash: {content_hash})")
                print(f"[PAGE SAMPLE] First 100 chars: '{text[:100]}'")
                return text
            else:
                print(f"[PAGE ERROR] No content div found for {page_id[:8]}...")
                # Log what divs we did find
                all_divs = soup.find_all("div")
                div_ids = [d.get("id") for d in all_divs if d.get("id")]
                print(f"[PAGE ERROR] Found divs with IDs: {div_ids}")
                # Log a sample of the HTML to understand the structure
                print(f"[PAGE ERROR] HTML sample: {response.text[:500]}")
        else:
            print(f"[PAGE ERROR] HTTP {response.status_code} for page {page_id[:8]}...")
            print(f"[PAGE ERROR] Response content preview: {response.text[:200]}")
    except Exception as e:
        print(f"[PAGE ERROR] Exception for page {page_id[:8]}...: {e}")
        import traceback
        print(f"[PAGE ERROR] Full traceback: {traceback.format_exc()}")
    return ""

def get_semantic_score_and_embedding(text: str):
    """
    Returns a (score, embedding) tuple.
    Enhanced scoring that combines multiple coherence signals.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        return 0.0, None

    try:
        text = text.strip()
        if not text:
            return 0.0, None

        # Basic text quality checks
        words = text.split()
        if len(words) < 3:
            return 0.0, None

        # Calculate basic coherence metrics
        alpha_ratio = sum(1 for w in words if w.isalpha()) / len(words)
        avg_word_len = sum(len(w.strip(",.!?;:\"'()[]{}")) for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)

        # If text is too incoherent, return low score
        if alpha_ratio < 0.3 or avg_word_len < 2:
            return 0.1, None

        text = text[:MAX_TEXT_LENGTH]  # limit length
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_EMBED_TOKENS)
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        sentence_embedding = token_embeddings.mean(dim=1).squeeze()

        # Get embedding norm (base semantic signal)
        norm_val = torch.norm(sentence_embedding).item()

        # Calculate variance across token embeddings (coherence signal)
        token_variance = torch.var(token_embeddings.squeeze(), dim=0).mean().item()

        # Combine signals with log scaling for better differentiation
        import math

        # Normalize and combine metrics
        norm_component = math.log(norm_val + 0.1) + 5  # Shift to positive range
        variance_component = math.log(token_variance + 0.001) + 10  # Coherence signal
        text_quality = alpha_ratio * unique_ratio * min(avg_word_len / 5, 1.0)

        # Final score with exponential scaling to amplify differences
        final_score = (norm_component + variance_component) * text_quality * 100

        # Ensure score is finite
        if math.isnan(final_score) or math.isinf(final_score):
            final_score = 0.0

        # Debug logging for scoring system - always log for now to debug the zero scores
        print(f"[SCORE DEBUG] ===================")
        print(f"[SCORE DEBUG] Input text: '{text[:100]}...'")
        print(f"[SCORE DEBUG] Text length: {len(text)}, Words: {len(words)}")
        print(f"[SCORE DEBUG] Alpha ratio: {alpha_ratio:.4f}, Avg word len: {avg_word_len:.4f}, Unique ratio: {unique_ratio:.4f}")
        print(f"[SCORE DEBUG] Embedding norm: {norm_val:.6f}")
        print(f"[SCORE DEBUG] Token variance: {token_variance:.8f}")
        print(f"[SCORE DEBUG] Norm component: {norm_component:.6f}")
        print(f"[SCORE DEBUG] Variance component: {variance_component:.6f}")
        print(f"[SCORE DEBUG] Text quality: {text_quality:.6f}")
        print(f"[SCORE DEBUG] Final score: {final_score:.6f}")
        print(f"[SCORE DEBUG] ===================")

        # If score is exactly 0, that's suspicious - let's investigate
        if final_score == 0.0:
            print(f"[SCORE WARNING] Score is exactly 0.0 - this might indicate a problem!")
            print(f"[SCORE WARNING] Check if text quality is 0: {text_quality}")
            print(f"[SCORE WARNING] Check if norm/variance components are problematic")

        return float(final_score), sentence_embedding

    except Exception as e:
        print(f"[SCORE ERROR] Error in scoring: {e}")
        return 0.0, None

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
    for item in pop:
        # Check if this is a test text (string) or a page ID
        if len(item) == PAGE_ID_LENGTH and all(c in '0123456789abcdef' for c in item):
            # This looks like a page ID
            full_text = get_page_text(item)
            page_id = item
        else:
            # This is test text
            full_text = item
            page_id = f"TEST_{hash(item) % 10000:04d}"

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
        self.run_count = 0

    def run(self):
        """Continually run small GA searches, store top hits in DB."""
        global status_data, evolution_history, generation_details

        # Resume from last run count
        self.run_count = status_data.get("current_run", 0)

        while self.running:
            # Clear memory before each run
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            self.run_count += 1
            status_data["current_run"] = self.run_count
            status_data["status"] = f"Running generation-based evolution (Run {self.run_count})"
            save_status_data(status_data)

            # TEST MODE: Use known good content to test scoring
            if self.run_count <= 5:  # Extend test mode to 5 runs
                print(f"[GA TEST] Running in TEST MODE for run {self.run_count}")
                # Use test content instead of random pages
                test_texts = [
                    "The quick brown fox jumps over the lazy dog in the meadow.",
                    "Once upon a time in a galaxy far away, there lived a wise old wizard.",
                    "To be or not to be, that is the question Shakespeare posed.",
                    "In the beginning was the Word, and the Word was with God.",
                    "It was the best of times, it was the worst of times in history.",
                    "Call me Ishmael, said the narrator of Moby Dick.",
                    "In a hole in the ground there lived a hobbit named Bilbo.",
                    "All happy families are alike, but every unhappy family differs.",
                    "It is a truth universally acknowledged that a single man must want a wife.",
                    "Last night I dreamt I went to Manderley again in my sleep."
                ]
                population = test_texts
            else:
                # Generate more diverse, realistic Library of Babel page IDs
                # Use random hex strings with better entropy distribution
                population = []
                for _ in range(POPULATION_SIZE):
                    # Generate truly random page IDs with mixed characters
                    page_id = ''.join(random.choices('0123456789abcdef', k=32))
                    # Ensure some variation by setting random positions to different values
                    chars = list(page_id)
                    for pos in random.sample(range(32), random.randint(5, 15)):
                        chars[pos] = random.choice('0123456789abcdef')
                    population.append(''.join(chars))

                print(f"[GA DEBUG] Generated diverse population: {[p[:8] + '...' for p in population[:3]]}")

            run_best_score = 0.0
            run_best_page = None

            # 2. For a few generations, refine
            for gen in range(NUM_GENERATIONS_PER_RUN):
                print(f"[GA DEBUG] Starting generation {gen + 1}/{NUM_GENERATIONS_PER_RUN} of run {self.run_count}")
                status_data["current_generation"] = gen + 1
                status_data["pages_evaluated"] = status_data.get("pages_evaluated", 0) + len(population)

                print(f"[GA DEBUG] Evaluating population of {len(population)} pages...")
                evaluated = evaluate_population(population)
                print(f"[GA DEBUG] Population evaluated. Best score: {max(evaluated, key=lambda x: x['score'])['score']:.4f}")

                # Track best in this generation
                gen_best = max(evaluated, key=lambda x: x["score"])
                if gen_best["score"] > run_best_score:
                    run_best_score = gen_best["score"]
                    run_best_page = gen_best["page_id"]
                    status_data["best_score_this_run"] = run_best_score
                    status_data["best_page_id"] = run_best_page

                # Add to evolution history with global generation counter
                global_generation = (self.run_count - 1) * NUM_GENERATIONS_PER_RUN + gen + 1
                evolution_history.append({
                    "generation": global_generation,
                    "score": gen_best["score"],
                    "run": self.run_count
                })

                # Add detailed generation information for diagnostics
                generation_details.append({
                    "run": self.run_count,
                    "generation": gen + 1,
                    "global_generation": global_generation,
                    "best_score": gen_best["score"],
                    "best_page_id": gen_best["page_id"],
                    "text_sample": gen_best["snippet"][:100] + "..." if len(gen_best["snippet"]) > 100 else gen_best["snippet"],
                    "population_size": len(evaluated),
                    "population_details": "\n".join([f"ID: {e['page_id'][:12]}... Score: {e['score']:.4f}" for e in sorted(evaluated, key=lambda x: x["score"], reverse=True)[:5]]) + f"\n... and {len(evaluated)-5} more" if len(evaluated) > 5 else "\n".join([f"ID: {e['page_id'][:12]}... Score: {e['score']:.4f}" for e in sorted(evaluated, key=lambda x: x["score"], reverse=True)]),
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Keep only last 200 points for better visualization
                if len(evolution_history) > 200:
                    evolution_history = evolution_history[-200:]

                # Keep only last 50 detailed generations
                if len(generation_details) > 50:
                    generation_details = generation_details[-50:]

                # Save status and history after each generation
                save_status_data(status_data)
                save_evolution_history(evolution_history)
                save_generation_details(generation_details)

                parents = select_parents(evaluated, KEEP_RATIO)
                new_pop = breed_population(parents, POPULATION_SIZE)
                mutated = [mutate(pid, MUTATION_RATE) for pid in new_pop]
                population = mutated

            # Evaluate final pop
            final_evaluated = evaluate_population(population)
            # Update leaderboard if any are good
            try_update_leaderboard(final_evaluated)

            status_data["status"] = f"Completed Run {self.run_count} - Sleeping before next run"
            save_status_data(status_data)
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
    global status_data
    while model is None or tokenizer is None:
        status_data["status"] = "Loading AI model..."
        save_status_data(status_data)
        time.sleep(1)
    status_data["status"] = "Model loaded - Starting evolution"
    save_status_data(status_data)
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