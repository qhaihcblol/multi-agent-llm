from dataclasses import dataclass


@dataclass
class GenerationRequest:
    model: str # model, e.g. "gpt-4o"
    system_prompt: str  # instruction
    prompt: str # input 
    temperature: float
    max_tokens: int 
