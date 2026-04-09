import logging
logger = logging.getLogger(__name__)

class VectorService:
    def faiss_build_index(self, embeddings):
        import faiss, numpy as np
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))
        return index

    def faiss_search(self, index, query_embedding, k=10):
        import numpy as np
        q = np.array([query_embedding], dtype="float32")
        distances, indices = index.search(q, k)
        return {"indices": indices[0].tolist(), "distances": distances[0].tolist()}

vector_service = VectorService()
