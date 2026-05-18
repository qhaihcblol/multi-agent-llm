from src.rag.components.embedder import Embedder
from src.rag.components.generator import Generator
from src.rag.components.prompt_builder import PromptBuilder
from src.rag.components.retriever import Retriever
from src.rag.entities.register_agent import RegisterAgent

from src.llm.clients.openai import OpenAIClient
from src.llm.configs.openai import OpenAIConfig


def main():
    embedder = Embedder(device="cpu")
    llm_client = OpenAIClient(config=OpenAIConfig.from_env())
    generator = Generator(llm_client=llm_client)