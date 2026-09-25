
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Page title
st.title("Semantic Search App")
st.write("Search the dataset using semantic similarity.")

# Load dataset
file_path = "Ex4_Dataset.txt"

with open(file_path, "r", encoding="utf-8") as file:
    documents = file.readlines()

documents = [doc.strip() for doc in documents if doc.strip()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create document embeddings
doc_embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Search box
query = st.text_input("Enter your search query:")

# Number of results
top_k = 3

if st.button("Search"):
    if query:
        # Convert query to embedding
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        # Search FAISS
        distances, indices = index.search(
            query_embedding,
            top_k
        )

        # Display results
        st.subheader("Search Results")

        for rank, idx in enumerate(indices[0]):
            st.write(f"### Rank {rank + 1}")
            st.write(documents[idx])
            st.write(f"Distance: {distances[0][rank]}")
    else:
        st.warning("Please enter a search query.")
