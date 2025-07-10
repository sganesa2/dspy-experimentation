# dspy-experimentation

This repository provides a modular, research-oriented framework for building, experimenting with, and evaluating Gen-AI-powered product recommendation systems using the [DSPy](https://github.com/stanfordnlp/dspy) library from the Stanford NLP Group. The codebase is designed for rapid prototyping and benchmarking of retrieval-augmented generation (RAG) pipelines, with a focus on product search and recommendation tasks. It integrates large language models (LLMs), a ColBERT-based retriever, and utilities for prompt optimization and evaluation.

## Features

- **End-to-End Product Recommendation**: Modular pipeline for retrieving, reasoning, and recommending products based on user queries.
- **Retrieval-Augmented Generation**: Uses a ColBERT-based retriever to fetch relevant products before LLM-based reasoning.
- **Prompt Optimization**: Supports few-shot and bootstrap teleprompting for optimizing prompts and system performance.
- **Evaluation Utilities**: Includes precision@k and other metrics, plus dataset utilities for training, testing, and development.
- **Jupyter Notebook**: Example notebook for interactive experimentation and pipeline optimization.

## Directory Structure

- `src/`: Main source code
  - `main.py`: Entry point (demo function)
  - `products.json`: Product catalog
  - `recsys_dataset.json`: Labeled dataset for training/evaluation
  - `utils/`: Core modules
    - `rec_system.py`: Recommendation system and dataset utilities
    - `retriever.py`: ColBERT retriever integration
    - `metrics.py`: Evaluation metrics
- `tests/`: Pytest-based unit tests
- `signature.ipynb`: Example notebook for DSPy pipeline and optimization

## Setup

1. **Clone the repository**  
   ```sh
   git clone <repo-url>
   cd dspy-experimentation
   ```

2. **Install dependencies**  
   It is recommended to use a virtual environment:
   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. **Configure environment variables**  
   Copy `.env.example` to `.env` and fill in your API keys:
   ```sh
   cp src/.env.example src/.env
   # Edit src/.env with your GROQ_API_KEY and model names
   ```

4. **Run the main script**  
   ```sh
   python src/main.py
   ```

5. **Run tests**  
   ```sh
   pytest
   ```

6. **Experiment in Jupyter Notebook**  
   Open `src/signature.ipynb` for interactive experimentation.

## Usage

- **Recommendation System**:  
  The main logic is in [`utils/rec_system.py`](src/utils/rec_system.py). Instantiate `RecommendationSystem` and call it with a product query.
- **Retriever**:  
  The ColBERT retriever is implemented in [`utils/retriever.py`](src/utils/retriever.py).
- **Evaluation**:  
  Precision@k is implemented in [`utils/metrics.py`](src/utils/metrics.py).

## Example

```python
from utils.rec_system import RecommendationSystem
recsys = RecommendationSystem(k=3)
result = recsys(product_query="I want a 2 in pvc pipe")
print(result)
```

## Notes

- Requires Python 3.11+
- You need valid keys for LLM access (see `.env.example`)
- Product data and datasets are in JSON files under `src/`

---
For more details, see the code in [src/](src) and the notebook [src/signature.ipynb](src/signature.ipynb).