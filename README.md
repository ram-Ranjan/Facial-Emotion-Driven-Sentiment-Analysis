# Facial Emotion Recognition & Review Generation System

## Overview
This project combines facial emotion recognition with synthetic review generation and semantic search capabilities using PyTorch, Transformers, and FAISS vector stores.

---

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create & Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train Facial Emotion Model

```bash
python model.py
```

Produces trained model + metrics in `outputs/`

### Generate Synthetic Reviews

```bash
python synthesis.py
```


**Features:**
- Upload image → detect emotion
- Generate review → embed → query
- View sentiment analysis

---

## Tech Stack

**Modeling:** PyTorch / Torchvision (ResNet-18)

**NLP & Sentiment:** HuggingFace Transformers, DistilBERT

**Language:** Python 3.12

---


## Future Improvements

- Add full 7-class dataset for higher accuracy
- Integrate LangChain RetrievalQA for production-scale RAG
- Replace rule-based reviews with GPT / T5 generation
- Dockerize + deploy via FastAPI

---

## Author

**Ranjan Kumar**  
M.Tech AI, NIT Silchar
