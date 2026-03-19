from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Seed text for generation")
    max_tokens: int = Field(50, ge=1, le=200, description="Number of tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(10, ge=0, le=50, description="Top-k candidates (0 = greedy)")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    parameters: dict
    tokens_generated: int
