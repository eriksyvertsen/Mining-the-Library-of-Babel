# Evolutionary Semantic Miner of the Library of Babel

## Overview

This application combines evolutionary computation with advanced semantic analysis to mine coherent text from the Library of Babel (libraryofbabel.info). The project employs genetic algorithms for exploration and sophisticated language models for semantic evaluation, searching for islands of meaning in an ocean of mathematical randomness.

The Library of Babel, inspired by Jorge Luis Borges' short story, contains every possible 3200-character page of text using 29 characters (lowercase letters, space, comma, and period). There exist exactly 25^1,312,000 books in the Library. If you wrote out this number in standard notation it would require about 1.8 million digits. Within this vastness lies every book ever written, every book that could ever exist, and an overwhelming ocean of gibberish. This project navigates this space using both evolutionary strategies and semantic intelligence.

## Core Components

### 1. Evolutionary Engine
The genetic algorithm explores the Library's vast address space by:
- Maintaining populations of page addresses
- Evolving addresses through mutation and crossover
- Selecting for semantic fitness
- Adapting search strategies based on discovered patterns

### 2. Semantic Evaluation System
The semantic analyzer determines text coherence through:
- **Language Model Perplexity**: Measuring how "surprised" GPT-2, BERT, or other models find the text
- **Syntactic Analysis**: Parsing for grammatical structures and sentence boundaries
- **Lexical Coherence**: Evaluating word choice patterns and vocabulary consistency
- **Semantic Density**: Calculating the ratio of meaningful content to random characters
- **Contextual Embedding Analysis**: Using transformer models to assess semantic relationships

## How It Works

### Phase 1: Initialization
```
Population → Random page addresses
         ↓
    Fetch pages from Library
         ↓
Initial semantic evaluation
```

### Phase 2: Semantic Analysis Pipeline
Each page undergoes multi-stage evaluation:

1. **Pre-processing**
   - Character frequency analysis
   - Basic structural detection (words, potential sentences)
   - Noise filtering

2. **Deep Semantic Evaluation**
   - Transformer model inference
   - Perplexity scoring across multiple models
   - Coherence metrics calculation
   - Semantic vector analysis

3. **Composite Scoring**
   - Weighted combination of all metrics
   - Normalization across population
   - Fitness assignment

### Phase 3: Evolution
```
High-scoring pages → Selection
                 ↓
            Reproduction
                 ↓
    Mutation & Crossover
                 ↓
         New generation
```

## Semantic Evaluation Details

### Language Models Employed

The system uses multiple models for robust evaluation:

- **GPT-2**: Primary perplexity scoring
- **BERT**: Masked language modeling for local coherence
- **RoBERTa**: Enhanced semantic understanding
- **Custom n-gram models**: Trained on various text corpora

### Coherence Metrics

1. **Perplexity Score**
   - Lower perplexity indicates more predictable, coherent text
   - Calculated using sliding windows across the page
   - Normalized against baseline random text

2. **Syntactic Validity**
   - Part-of-speech tagging success rate
   - Dependency parsing completeness
   - Sentence boundary detection accuracy

3. **Semantic Consistency**
   - Word embedding coherence across the page
   - Topic modeling stability
   - Named entity recognition patterns

4. **Information Content**
   - Entropy calculations
   - Compression ratio analysis
   - Meaningful word density

### Scoring Algorithm

```python
def calculate_semantic_fitness(page_text):
    # Multi-model perplexity
    gpt2_score = gpt2_model.evaluate_perplexity(page_text)
    bert_score = bert_model.evaluate_coherence(page_text)

    # Syntactic analysis
    syntax_score = syntactic_analyzer.parse_validity(page_text)

    # Semantic density
    density = semantic_analyzer.calculate_density(page_text)

    # Weighted combination
    fitness = (
        0.35 * normalize(gpt2_score) +
        0.25 * normalize(bert_score) +
        0.20 * syntax_score +
        0.20 * density
    )

    return fitness
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for transformer models)
- 16GB+ RAM for model loading
- Internet connection for Library of Babel access

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/babel-semantic-miner.git
cd babel-semantic-miner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download language models
python download_models.py
```

### Required Dependencies

```
# Core
requests>=2.28.0
numpy>=1.24.0
scipy>=1.10.0

# Language Models
transformers>=4.30.0
torch>=2.0.0
tokenizers>=0.13.0

# NLP Tools
nltk>=3.8.0
spacy>=3.5.0
scikit-learn>=1.2.0

# Utilities
beautifulsoup4>=4.12.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=2.0.0
```

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Configuration

```bash
# High-performance semantic mining
python main.py \
    --population-size 200 \
    --generations 100 \
    --mutation-rate 0.1 \
    --semantic-models gpt2,bert,roberta \
    --evaluation-depth deep \
    --parallel-workers 4
```

### Command Line Arguments

**Evolution Parameters:**
- `--population-size`: Number of page IDs per generation (default: 50)
- `--generations`: Number of evolution cycles (default: 100)
- `--mutation-rate`: Probability of mutation per character (default: 0.05)
- `--crossover-rate`: Probability of crossover (default: 0.7)
- `--elite-size`: Number of top performers to preserve (default: 10)

**Semantic Evaluation Parameters:**
- `--semantic-models`: Comma-separated list of models (default: 'gpt2')
- `--evaluation-depth`: 'shallow', 'normal', or 'deep' (default: 'normal')
- `--perplexity-threshold`: Maximum acceptable perplexity (default: 1000)
- `--coherence-window`: Size of sliding window for analysis (default: 512)
- `--batch-size`: Pages to evaluate simultaneously (default: 8)

**System Parameters:**
- `--parallel-workers`: Number of parallel evaluation threads (default: 2)
- `--cache-size`: Maximum cached pages (default: 10000)
- `--output`: Directory for results (default: './results')
- `--checkpoint-interval`: Generations between saves (default: 10)

## Project Structure

```
babel-semantic-miner/
├── main.py                    # Main orchestrator
├── evolution/
│   ├── genetic_algorithm.py   # GA implementation
│   ├── population.py          # Population management
│   ├── operators.py           # Mutation/crossover
│   └── selection.py           # Selection strategies
├── semantic/
│   ├── evaluator.py           # Main semantic evaluation
│   ├── language_models.py     # Model interfaces
│   ├── metrics.py             # Coherence metrics
│   ├── preprocessor.py        # Text preprocessing
│   └── analyzers/
│       ├── syntactic.py       # Syntax analysis
│       ├── lexical.py         # Word-level analysis
│       └── contextual.py      # Deep semantic analysis
├── babel/
│   ├── interface.py           # Library of Babel API
│   ├── page_cache.py          # Caching system
│   └── address_manager.py     # Address encoding/decoding
├── utils/
│   ├── visualization.py       # Progress/result plotting
│   ├── logging.py             # Advanced logging
│   └── checkpoint.py          # State saving/loading
├── models/                    # Pretrained model storage
├── config/
│   ├── default.json           # Default configuration
│   └── models.json            # Model configurations
├── results/                   # Output directory
└── tests/                     # Comprehensive test suite
```

## Configuration

### Semantic Evaluation Configuration

```json
{
  "semantic": {
    "models": {
      "gpt2": {
        "enabled": true,
        "weight": 0.35,
        "max_length": 1024,
        "stride": 512
      },
      "bert": {
        "enabled": true,
        "weight": 0.25,
        "mask_ratio": 0.15
      },
      "roberta": {
        "enabled": false,
        "weight": 0.20
      }
    },
    "metrics": {
      "perplexity_threshold": 1000,
      "min_word_ratio": 0.6,
      "syntactic_threshold": 0.4
    },
    "preprocessing": {
      "remove_excess_spaces": true,
      "normalize_punctuation": true,
      "min_token_length": 2
    }
  },
  "evolution": {
    "population_size": 50,
    "mutation_rate": 0.05,
    "crossover_rate": 0.7,
    "tournament_size": 3,
    "elite_percentage": 0.2
  }
}
```

## Understanding Results

### Output Files

1. **evolution_log.csv**: Generation-by-generation statistics
2. **best_pages.json**: Top-scoring pages with full text and metrics
3. **semantic_analysis/**: Detailed analysis for each high-scoring page
4. **visualizations/**: Fitness progression graphs and semantic maps

### Interpreting Semantic Scores

- **Score 0-0.2**: Random gibberish
- **Score 0.2-0.4**: Occasional word-like patterns
- **Score 0.4-0.6**: Partial sentences, recognizable fragments
- **Score 0.6-0.8**: Coherent phrases, near-meaningful text
- **Score 0.8-1.0**: Genuinely coherent passages

## Advanced Features

### Semantic Targeting
Guide evolution toward specific semantic goals:
```bash
python main.py --semantic-target "scientific prose" --target-weight 0.3
```

### Multi-objective Optimization
Balance multiple semantic qualities:
```bash
python main.py --objectives coherence,creativity,specificity
```

### Distributed Processing
Run across multiple machines:
```bash
python main.py --mode distributed --coordinator-url http://main-node:8080
```

## Performance Optimization

### GPU Acceleration
- Models automatically use CUDA if available
- Batch processing optimizes GPU utilization
- Mixed precision training reduces memory usage

### Caching Strategy
- LRU cache for frequently accessed pages
- Semantic evaluation results cached
- Persistent cache across runs

### Parallel Processing
- Concurrent page fetching
- Parallel semantic evaluation
- Asynchronous evolution operations

## Research Applications

This tool enables research in:
- **Computational Linguistics**: Studying coherence emergence
- **Information Theory**: Analyzing meaning in random systems
- **Evolutionary Computation**: Testing GA performance in extreme search spaces
- **Philosophy of Language**: Exploring meaning and randomness boundaries

## Limitations and Considerations

- **Computational Intensity**: Deep semantic evaluation requires significant resources
- **API Constraints**: Library of Babel rate limits may slow exploration
- **Model Biases**: Language models may favor certain text styles
- **Local Optima**: Evolution may converge on semi-coherent patterns

## Contributing

We welcome contributions in:
- Additional semantic evaluation metrics
- Alternative evolutionary strategies
- Performance optimizations
- Model integration
- Result visualization

Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Jorge Luis Borges for the conceptual foundation
- Jonathan Basile for libraryofbabel.info
- Hugging Face for transformer model infrastructure
- The evolutionary computation research community

## Citation

```bibtex
@software{babel_semantic_miner,
  title = {Evolutionary Semantic Miner of the Library of Babel},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/babel-semantic-miner},
  note = {A hybrid evolutionary-semantic system for coherent text discovery}
}
```

---

*"The certitude that everything has been written negates us or turns us into phantoms."* - Jorge Luis Borges