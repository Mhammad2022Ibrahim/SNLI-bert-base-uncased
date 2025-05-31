# SNLI BERT Base Uncased

This repository contains a BERT-base-uncased model fine-tuned for the Stanford Natural Language Inference (SNLI) task. It provides a simple FastAPI-based REST API for sequence classification to predict relationships between premise-hypothesis pairs: entailment, contradiction, and neutral.

---

## Model

- Pretrained BERT base uncased (`bert-base-uncased`) fine-tuned on the SNLI dataset.
- Model repository on Hugging Face: [Mhammad2023/snli-bert-base-uncased](https://huggingface.co/Mhammad2023/snli-bert-base-uncased)
- Dataset: [Stanford SNLI Dataset](https://huggingface.co/datasets/stanfordnlp/snli)

---

## Features

- Predicts three classes:
  - `contradiction`
  - `entailment`
  - `neutral`
- Returns the predicted label with confidence score.
- FastAPI based asynchronous API for easy integration and deployment.

---

## Installation

Make sure you have Python 3.8+ installed. Then create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```
## Note: Ensure transformers, torch, and fastapi are installed.

## Usage
Run the API server locally
```bash
uvicorn app.main:app --reload
```

## Test the API
Send a POST request to /predict endpoint with JSON payload:
```json
{
  "premise": "A man inspects the uniform of a figure in some East Asian country.",
  "hypothesis": "The man is sleeping."
}```

Expected response:
```json
{
  "premise": "A man inspects the uniform of a figure in some East Asian country.",
  "hypothesis": "The man is sleeping.",
  "label": "contradiction",
  "confidence": 0.98
}```