import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM Inference API", version="1.0.0")


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7


class InferenceResponse(BaseModel):
    response: str
    tokens_used: int
    inference_time: float
    model: str


class HealthResponse(BaseModel):
    status: str
    timestamp: float


@app.get("/", response_model=dict)
async def root():
    return {"message": "LLM Inference API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", timestamp=time.time())


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Mock LLM inference endpoint
    In production, this would call an actual LLM model
    """
    start_time = time.time()

    # Validate input
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if request.max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")

    # Mock inference
    mock_response = f"This is a mock response to: {request.prompt[:50]}..."
    tokens_used = min(request.max_tokens, len(mock_response.split()))

    inference_time = time.time() - start_time

    return InferenceResponse(
        response=mock_response,
        tokens_used=tokens_used,
        inference_time=inference_time,
        model="mock-llm-v1",
    )
