#############################
# main.py for Replit
# Updated GA "Mining" UI with interpretability
#############################

import requests
import random
import torch
import time
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, render_template_string

#######################
# CONFIGURATIONS
#######################
POPULATION_SIZE = 10     # Fewer for quick tests
NUM_GENERATIONS = 3      # Fewer gens so it finishes faster
KEEP_RATIO = 0.5
MUTATION_RATE = 0.02
PAGE_ID_LENGTH = 32      # Typical hex ID length on libraryofbabel.info

REQUEST_TIMEOUT = 10      # seconds
MAX_TEXT_LENGTH = 1000    # how many chars we read from each page
MAX_EMBED_TOKENS = 256    # how many tokens we embed at once

#######################
# LOAD MODEL
#######################
print("Loading model. This may take a while on Replit...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()
print("Model loaded.")

#######################
# FLASK APP
#######################
app = Flask(__name__)

# -- HTML TEMPLATE --
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Project 2150: Distributed Semantic Mining</title>
    <style>
        body {
            background: #000000;
            color: #00ff99;
            font-family: "Consolas", monospace;
            margin: 0; 
            padding: 0;
        }
        #header {
            background: linear-gradient(to right, #003300, #001a00);
            padding: 20px;
            text-align: center;
        }
        #header h1 {
            margin: 0;
            font-size: 2em;
        }
        #content {
            margin: 20px auto;
            width: 90%;
            max-width: 900px;
            background: rgba(0, 128, 64, 0.1);
            padding: 20px;
            border: 1px solid #00ff99;
            border-radius: 8px;
        }
        button {
            background-color: #00ff99;
            color: #000;
            border: none;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
            text-transform: uppercase;
            font-weight: bold;
        }
        button:hover {
            background-color: #33ffaa;
        }
        pre {
            background: #001f0f;
            padding: 10px;
            color: #00ff99;
            border: 1px solid #00ff99;
            border-radius: 4px;
            white-space: pre-wrap; 
            word-wrap: break-word;
        }
        a {
            color: #00ff99;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>PROJECT 2150 - DISTRIBUTED SEMANTIC MINING</h1>
        <p>Year 2150: Global collective harnessing quantum networks to find meaning in the Library of Babel.</p>
    </div>

    <div id="content">
        <h2>Genetic Algorithm Mining Interface (with Interpretability)</h2>
        <p>Initiate an evolutionary search across randomly generated pages in the Library of Babel. 
           We attempt to find <em>islands of coherent text</em> hidden among oceans of randomness.</p>
        <p><strong>Note</strong>: This can be slow and may not find meaningful text quickly.</p>

        <form method="POST" action="/start">
            <button type="submit">Begin Mining</button>
        </form>

        {% if logs %}
        <hr>
        <h3>Mining Log</h3>
        <pre>{{ logs }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """
    Home page: show the futuristic interface.
    """
    return render_template_string(HTML_TEMPLATE, logs=None)


@app.route("/start", methods=["POST"])
def start():
    """
    Trigger the genetic algorithm, then display logs and results.
    """
    logs = run_genetic_algorithm()
    return render_template_string(HTML_TEMPLATE, logs=logs)


##############################
# GA + SCRAPING LOGIC
##############################
def get_page_text(page_id: str) -> str:
    """
    Retrieves the text from a Library of Babel page given its page_id.
    """
    url = f"https://libraryofbabel.info/book.cgi?{page_id}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_content_div = soup.find("div", {"id": "page"})
            if page_content_div:
                return page_content_div.get_text()
        return ""
    except:
        return ""

def get_semantic_score_and_embedding(text: str):
    """
    Returns:
      - semantic_score (float): numeric measure of coherence
      - embedding (torch.Tensor): the final sentence embedding
    """
    text = text.strip()
    if not text:
        return 0.0, None

    # Keep only first N chars
    text = text[:MAX_TEXT_LENGTH]

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_EMBED_TOKENS)
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    sentence_embedding = token_embeddings.mean(dim=1).squeeze()  # shape: (768,)
    norm_val = torch.norm(sentence_embedding).item()
    return float(norm_val), sentence_embedding

def interpret_text(snippet: str, embedding_norm: float):
    """
    Provide basic interpretability metrics for a text snippet.
    We'll show:
      - The embedding norm (already known from get_semantic_score_and_embedding).
      - Total token count (words).
      - Avg word length.
      - Ratio of alphabetic tokens.
      - Repeated word fraction.
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

    # Split on whitespace to get tokens
    tokens = snippet.split()
    token_count = len(tokens)
    if token_count == 0:
        return {
            "Token count": 0,
            "Avg word length": 0.0,
            "Alphabetic ratio": 0.0,
            "Repeated word ratio": 0.0,
            "Embedding norm": embedding_norm
        }

    # Clean tokens (remove punctuation, etc.) for analysis
    cleaned_tokens = [t.strip(",.!?;:\"'()[]{}") for t in tokens]
    # Calculate average length
    avg_len = sum(len(t) for t in cleaned_tokens) / token_count

    # Ratio of alphabetic tokens
    alpha_tokens = [t for t in cleaned_tokens if t.isalpha()]
    alpha_ratio = len(alpha_tokens) / token_count

    # Repeated word ratio
    unique_tokens = set(cleaned_tokens)
    repeated_count = token_count - len(unique_tokens)
    repeated_ratio = repeated_count / token_count if token_count else 0

    return {
        "Token count": token_count,
        "Avg word length": round(avg_len, 2),
        "Alphabetic ratio": round(alpha_ratio, 2),
        "Repeated word ratio": round(repeated_ratio, 2),
        "Embedding norm": round(embedding_norm, 3)
    }

def random_page_id(length=PAGE_ID_LENGTH):
    return ''.join(random.choice('0123456789abcdef') for _ in range(length))

def mutate(page_id, mutation_rate=MUTATION_RATE):
    page_chars = list(page_id)
    for i in range(len(page_chars)):
        if random.random() < mutation_rate:
            possible_digits = [d for d in '0123456789abcdef' if d != page_chars[i]]
            page_chars[i] = random.choice(possible_digits)
    return ''.join(page_chars)

def crossover(parent1_id, parent2_id):
    length = len(parent1_id)
    cross_point = length // 2
    child_id = parent1_id[:cross_point] + parent2_id[cross_point:]
    return child_id

def evaluate_population(pop):
    """
    Return a list of dictionaries:
      {
        'page_id': ...,
        'snippet': ...,
        'score': ...,
        'metrics': {... interpretability metrics ...}
      }
    """
    results = []
    for pid in pop:
        text = get_page_text(pid)
        score, embedding = get_semantic_score_and_embedding(text)
        snippet = text[:300].replace("\n", " ")  # short snippet

        # Build interpretability metrics from snippet + embedding norm
        metrics = interpret_text(snippet, score)

        item = {
            'page_id': pid,
            'score': score,
            'snippet': snippet,
            'metrics': metrics
        }
        results.append(item)
    return results

def select_parents(evaluated, keep_ratio=KEEP_RATIO):
    sorted_eval = sorted(evaluated, key=lambda x: x['score'], reverse=True)
    keep_count = max(1, int(len(sorted_eval)*keep_ratio))
    return sorted_eval[:keep_count]

def breed_population(parents, new_size):
    parent_ids = [p['page_id'] for p in parents]
    new_pop = []
    while len(new_pop) < new_size:
        p1 = random.choice(parent_ids)
        p2 = random.choice(parent_ids)
        child = crossover(p1, p2)
        new_pop.append(child)
    return new_pop

def maintain_diversity(population, target_size):
    unique_inds = list(set(population))
    if len(unique_inds) < target_size * 0.8:
        while len(unique_inds) < target_size:
            unique_inds.append(random_page_id())
    return unique_inds

def run_genetic_algorithm():
    """
    Executes the GA, returns a multi-line string of logs.
    """
    log_lines = []
    population = [random_page_id() for _ in range(POPULATION_SIZE)]

    for gen in range(NUM_GENERATIONS):
        log_lines.append(f"=== Generation {gen} ===")
        evaluated = evaluate_population(population)
        sorted_eval = sorted(evaluated, key=lambda x: x['score'], reverse=True)
        best_item = sorted_eval[0]
        log_lines.append(f"Best so far => {best_item['page_id']} [Score: {best_item['score']:.4f}]")
        log_lines.append(f"Snippet: {best_item['snippet']}")
        log_lines.append(f"Metrics: {best_item['metrics']}\n")

        parents = select_parents(evaluated, KEEP_RATIO)
        new_population = breed_population(parents, POPULATION_SIZE)
        mutated_population = [mutate(pid, MUTATION_RATE) for pid in new_population]
        population = maintain_diversity(mutated_population, POPULATION_SIZE)

    # Final results
    final_eval = evaluate_population(population)
    final_sorted = sorted(final_eval, key=lambda x: x['score'], reverse=True)
    log_lines.append("\n=== FINAL TOP 5 ===")
    for i in range(min(5, len(final_sorted))):
        item = final_sorted[i]
        pid = item['page_id']
        sc = item['score']
        snippet = item['snippet']
        url = f"https://libraryofbabel.info/book.cgi?{pid}"
        metrics = item['metrics']
        log_lines.append(
            f"Rank {i+1}: {pid} (Score: {sc:.4f})\n"
            f"Snippet: {snippet}\n"
            f"Metrics: {metrics}\n"
            f"URL: {url}\n"
        )

    return "\n".join(log_lines)

#################
# FLASK ENTRY
#################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
