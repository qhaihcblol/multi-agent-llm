from .ingestor import Ingestor
from .query_engine import QueryEngine


class Agent:
    def __init__(self, ingestor: Ingestor, query_engine: QueryEngine, doc_path: str):
        self.ingestor = ingestor
        self.query_engine = query_engine
        # Add later