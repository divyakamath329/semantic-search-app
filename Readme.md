# 🔎 Semantic Search App

A simple Semantic Search application built using **Python, Streamlit, Sentence Transformers, and FAISS**.

## 📌 Project Overview

This project demonstrates how semantic search can find relevant information based on the **meaning of a query**, rather than only matching exact keywords.

The application converts the documents and the user's query into numerical **embeddings** and then uses FAISS to find the most similar documents.

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- NumPy
- `all-MiniLM-L6-v2` embedding model

## ⚙️ How It Works

1. The application reads documents from `Ex4_Dataset.txt`.
2. The documents are converted into embeddings using Sentence Transformers.
3. FAISS creates a vector index for the document embeddings.
4. The user enters a search query.
5. The query is converted into an embedding.
6. FAISS compares the query embedding with the document embeddings.
7. The top 3 most relevant results are displayed.

## 📂 Project Structure

```text
semantic-search-app/
│
├── app.py
├── Ex4_Dataset.txt
├── requirements.txt
└── README.md

