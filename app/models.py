from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model_name: str = Field(default="mock-llm-v1")
    max_context_length: int = Field(default=2048)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=100, ge=1, le=2048)


class ModelMetrics(BaseModel):
    total_requests: int = 0
    average_inference_time: float = 0.0
    total_tokens_generated: int = 0
