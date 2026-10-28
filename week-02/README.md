# Week 02 - Mini Semantic Search System

**Objective**

Build a mini semantic search system that loads FAQ data from CSV/JSON using pandas, cleans and splits the text into chunks, converts text into embeddings, stores them in ChromaDB, and performs semantic search to display the top 3 most relevant matches.
# Google colab link 
https://colab.research.google.com/drive/16WtUfPZwJa6r-mbQk_mXeGHofpNzEmrI#scrollTo=cd25ff6d

**What the task covers**

| Step | What it does | Tool |
| --- | --- | --- |
| 1. Load & inspect | Read FAQs from CSV or JSON | pandas |
| 2. Clean | Clean text and prepare chunks | pandas/numpy |
| 3. Embeddings | Generate vector embeddings from text | sentence-transformers |
| 4. Store | Store embeddings and metadata in vector DB | chromadb |
| 5. Search | Query with text and retrieve top 3 matches | chromadb/sentence-transformers |

**Files**

- semantic_search_system.py - Standalone script to build the index and run sample search.
- semantic_search_system.ipynb - Jupyter/Colab notebook with outputs and explanations.
- data/faqs.csv, data/faqs.json - Sample FAQ datasets.
- 
equirements.txt - Packages needed to run.

**How to run**

`ash
pip install -r requirements.txt
python semantic_search_system.py
``n
**Usage**

The script loads FAQs, creates chunks (question+answer), generates embeddings, stores them in ChromaDB, runs a sample query, and prints the top 3 most relevant results.
