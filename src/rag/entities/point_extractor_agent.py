
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..schemas.point import Point
from .node import Node


class PointExtractorAgent:
    def __init__(self, generator: Generator, prompt_builder: PromptBuilder) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder
        
    def extract_points(self, nodes: list[Node], question: str, top_k: int = 5):
        for node in nodes:
            retrieved_chunks = node.retrieve(query=question, top_k=top_k)
            