"""Initialize ChromaDB after pip install -r requirements.txt."""
from app.retrieval import index_catalog

if __name__ == "__main__":
    index_catalog()
