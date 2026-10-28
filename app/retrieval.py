"""Small, genuine ChromaDB semantic RAG over the 18 catalogue records."""
from pathlib import Path

from app.builder import load_catalog

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chroma_db"
COLLECTION = "otraverse_catalog_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def catalogue_documents(catalog: dict) -> list[dict]:
    """One indexed document per template/environment/asset/rule: 3+3+7+5=18."""
    docs = []
    for category, key, singular in (
        ("game_template", "game_templates", "game template"),
        ("environment", "environments", "game environment"),
        ("asset", "assets", "game asset"),
    ):
        for entry in catalog[key]:
            docs.append({
                "id": f"{category}:{entry['id']}",
                "text": f"Supported {singular}: {entry['name']}. Catalogue ID: {entry['id']}.",
                "metadata": {"category": category, "catalog_id": entry["id"], "source": "catalog.json"},
            })
    for key, limits in catalog["rules"].items():
        if isinstance(limits, dict):
            description = f"Minimum {limits['min']}; maximum {limits['max']}"
        else:
            description = "Allowed values: " + ", ".join(limits)
        docs.append({
            "id": f"rule:{key}",
            "text": f"Game rule {key}: {description}.",
            "metadata": {"category": "rule", "catalog_id": key, "source": "catalog.json"},
        })
    return docs


def index_catalog():
    """Create/update the persistent vector index. Run once before starting the API."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    docs = catalogue_documents(load_catalog())
    if len(docs) != 18:
        raise ValueError(f"Expected 18 catalogue documents, got {len(docs)}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
        embeddings=vectors.tolist(),
    )
    assert {d["id"] for d in docs}.issubset(set(collection.get(include=[])["ids"]))
    print(f"Indexed {collection.count()} catalogue documents in {DB_PATH}")


class Retriever:
    def __init__(self):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(DB_PATH))
        try:
            self.collection = client.get_collection(name=COLLECTION)
        except Exception as exc:
            raise RuntimeError("Missing catalogue index. Run: python -m scripts.index") from exc

    def search(self, prompt: str, k: int = 5, category: str | None = None) -> list[dict]:
        if k <= 0:
            return []
        vector = self.model.encode(prompt, normalize_embeddings=True).tolist()
        where = {"category": category} if category else None
        count = self.collection.count()
        if count == 0:
            return []
        query = self.collection.query(
            query_embeddings=[vector], n_results=min(k, count),
            where=where, include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": doc_id,
                "text": query["documents"][0][i],
                "metadata": query["metadatas"][0][i],
                "distance": float(query["distances"][0][i]),
            }
            for i, doc_id in enumerate(query["ids"][0])
        ]

    def full_prompt(self, prompt: str, k: int = 5) -> list[dict]:
        return self.search(prompt, k=k)

    def category_aware(self, prompt: str, *, evaluation: bool = False) -> list[dict]:
        # Evaluation uses same 5-document budget as full_prompt.
        # Runtime includes rules and more assets to help complex prompts.
        budget = (
            {"game_template": 1, "environment": 1, "asset": 3}
            if evaluation else
            {"game_template": 1, "environment": 1, "asset": 5, "rule": 2}
        )
        return [
            item for category, k in budget.items()
            for item in self.search(prompt, k=k, category=category)
        ]
