# Text Analysis and Recommendation API 🧠💬

A sentiment analysis, linguistic feature extraction, and context-aware recommendation system. It integrates GPT-2, DistilBERT, and RoBERTa models.

Note: I used gpt2 (for money problem) but you must use 3 and latest version for better solutions.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<img width="1470" alt="Image" src="https://github.com/user-attachments/assets/30c4fc2a-318b-4f2b-83f8-e883eb5219c9" />
<img width="1465" alt="Image" src="https://github.com/user-attachments/assets/4a5b7162-1577-4dc3-b88a-2bb2a432c315" />

## Key Features ✨

- **Multi-Model Integration**: Synchronization of 3 different NLP models
- **Linguistic Analysis**:
  - 50+ idiom detection
  - Irony pattern matching
  - Context inference
- **Dynamic Scenario Detection**: 6 different context categories
- **GPT-2 Based Recommendation System**: Context-aware suggestions

## Tech Stack 🛠️

| Component            | Technologies                          |
|----------------------|---------------------------------------|
| **Backend Framework**| FastAPI, Uvicorn                      |
| **NLP Models**       | GPT-2-large, DistilBERT, GoEmotions-RoBERTa |
| **Hardware Optimization** | CUDA, MPS, CPU fallback           |
| **Processors**       | Regex, N-gram analysis                 |

## Setup 🚀

1. Clone the repository:
```bash
git clone https://github.com/nilayduman/dear_teddy_api.git
cd api_as

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt

API_PORT=8000
MAX_TEXT_LENGTH=512

uvicorn main:app --reload --port $API_PORT



