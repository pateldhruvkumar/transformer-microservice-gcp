import logging

from fastapi import APIRouter, HTTPException

from app.model import basic_english_tokenize, generate
from app.schemas import GenerateRequest, GenerateResponse
from app.state import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {
        "service": "Shakespeare Text Generator",
        "model": "Transformer Language Model (Tiny Shakespeare)",
        "usage": "POST /generate with JSON body",
        "docs": "/docs",
        "status": "running",
    }


@router.get("/health")
async def health():
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    logger.info(
        "Generate — prompt=%r max_tokens=%d temperature=%.2f top_k=%d",
        request.prompt, request.max_tokens, request.temperature, request.top_k,
    )

    try:
        result = generate(
            model=state.model,
            vocab=state.vocab,
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            device=state.device,
        )
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    prompt_token_count = len(basic_english_tokenize(request.prompt))
    tokens_generated = max(0, len(result.split()) - prompt_token_count)

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=result,
        parameters={
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
        },
        tokens_generated=tokens_generated,
    )
