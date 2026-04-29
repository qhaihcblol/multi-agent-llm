from ..components import Chunker, Embedder, VectorStore
class Ingestor:
    def __init__(self, chunker: Chunker, embedding: Embedder, vector_store: VectorStore):
        pass
    def load_document(self, doc_path: str):
        pass
    def chunk(self, document):
        pass
    def embed(self, chunk):
        pass
    def index(self, embedding):
        pass
    
    # Main method to ingest a document  
    def ingest(self, doc_path: str):
        pass