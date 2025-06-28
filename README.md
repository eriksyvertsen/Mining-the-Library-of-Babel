# Evolutionary Semantic Miner of the Library of Babel

## Overview

This application mines the infinite expanse of the Library of Babel (libraryofbabel.info) for coherent text using genetic algorithms and machine learning. The project searches for meaning in the mathematical near-infinity of random text combinations.

The Library of Babel, inspired by Jorge Luis Borges' short story, contains every possible 3200-character page of text using 29 characters (lowercase letters, space, comma, and period). There are exactly 25^1,312,000 books in the Library. If you wrote out this number in standard notation it would require about 1.8 million digits. Within this vastness lies every book ever written, every book that could ever exist, and an overwhelming ocean of gibberish. This project attempts to navigate this space intelligently.

## How It Works

### Core Algorithm

1. **Population Initialization**: The system generates a random population of page IDs
2. **Fitness Evaluation**: Each page gets retrieved from the Library and scored for coherence using a language model
3. **Selection**: The algorithm keeps the highest-scoring pages as "parents"
4. **Mutation**: Parent IDs undergo controlled mutations to create offspring
5. **Evolution**: The process repeats, gradually discovering pages with increasing coherence

### The Search Space

The Library of Babel uses a unique addressing system:
- **Hexagon**: A location identifier (up to 3260 characters)
- **Wall**: 1-4
- **Shelf**: 1-5  
- **Volume**: 1-32
- **Page**: 1-410

Each combination produces a unique page of pseudo-random text, deterministically generated from the address.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for Library of Babel API access

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
```

### Required Dependencies

```
requests>=2.28.0
numpy>=1.24.0
transformers>=4.30.0
torch>=2.0.0
beautifulsoup4>=4.12.0
tqdm>=4.65.0
```

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Configuration

```bash
python main.py --population-size 100 --generations 50 --mutation-rate 0.1
```

### Command Line Arguments

- `--population-size`: Number of page IDs per generation (default: 50)
- `--generations`: Number of evolution cycles (default: 100)
- `--mutation-rate`: Probability of mutation per character (default: 0.05)
- `--elite-size`: Number of top performers to preserve (default: 10)
- `--model`: Language model for scoring (default: 'gpt2')
- `--output`: Directory for saving results (default: './results')

## Technical Implementation

### Genetic Algorithm Components

#### Chromosome Representation
Page IDs encode as strings containing the hexagon location and page coordinates. The genetic algorithm treats these strings as chromosomes.

#### Fitness Function
The fitness function evaluates page coherence using:
- **Perplexity scores** from language models
- **Word frequency analysis** against known English distributions
- **N-gram probability** calculations
- **Sentence structure detection**

#### Mutation Strategies
1. **Point Mutation**: Random character changes in the hexagon string
2. **Coordinate Mutation**: Adjustments to wall/shelf/volume/page numbers
3. **Crossover**: Combining parts of two parent IDs
4. **Adaptive Mutation**: Rate changes based on fitness plateau detection

### Library of Babel Integration

The project interfaces with the Library through:
- Direct API calls to libraryofbabel.info
- HTML parsing for page content extraction
- Rate limiting to respect server resources
- Caching mechanisms for previously fetched pages

## Project Structure

```
project/
├── main.py       # Main execution script
├── genetic_algorithm.py   # GA implementation
├── babel_interface.py     # Library of Babel API wrapper
├── fitness_evaluator.py   # Text coherence scoring
├── utils/
│   ├── text_analysis.py   # NLP utilities
│   ├── caching.py         # Page cache management
│   └── visualization.py   # Progress plotting
├── models/
│   └── scoring_models.py  # Language model interfaces
├── results/               # Output directory
└── tests/                 # Unit tests
```

## Configuration File

Create a `config.json` for persistent settings:

```json
{
  "algorithm": {
    "population_size": 50,
    "mutation_rate": 0.05,
    "crossover_rate": 0.7,
    "elite_percentage": 0.2
  },
  "scoring": {
    "model": "gpt2",
    "weights": {
      "perplexity": 0.4,
      "word_frequency": 0.3,
      "grammar": 0.3
    }
  },
  "babel": {
    "rate_limit": 1.0,
    "timeout": 30,
    "cache_size": 10000
  }
}
```

## Example Results

After running for several generations, the Evolutionary Semantic Miner has discovered pages containing:
- Partial dictionary entries
- Fragments resembling poetry
- Statistical noise approaching readable sentences
- Occasionally, complete coherent phrases

## Performance Considerations

- **API Rate Limiting**: The Library of Babel may throttle requests. The system includes automatic backoff
- **Computational Cost**: Language model scoring requires significant CPU/GPU resources
- **Memory Usage**: Page caching can consume substantial RAM with large populations
- **Convergence Time**: Meaningful results typically emerge after 50+ generations

## Future Enhancements

### Planned Features
- Distributed computing support for larger populations
- Community sharing of discovered coherent pages

### Research Directions
- Investigating emergence patterns in the fitness landscape
- Analyzing the mathematical properties of coherent page clusters
- Developing specialized mutation operators for linguistic structures
- Creating guided search strategies based on semantic targets

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project operates under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- Jorge Luis Borges for the original "Library of Babel" concept
- Jonathan Basile for creating libraryofbabel.info
- The open-source NLP community for language modeling tools

## Citation

If you use this Evolutionary Semantic Miner in research, please cite:

```bibtex
@software{babel_semantic_miner,
  title = {Evolutionary Semantic Miner of the Library of Babel},
  year = {2025},
  url = {https://github.com/yourusername/babel-semantic-miner}
}
}
```

## Contact

For questions, suggestions, or discussions about discovering meaningful text in mathematical infinities, please open an issue on GitHub.

---

*"The universe (which others call the Library) composes all books."* - Jorge Luis Borges