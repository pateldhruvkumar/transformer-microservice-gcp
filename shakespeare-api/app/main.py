"""
main.py — FastAPI application factory.

Handles startup (artifact loading) and wires middleware + routes.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import TransformerModel, Vocab
from app.routes import router
from app.state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).parent.parent / "artifacts"))

MODEL_PATH      = ARTIFACTS_DIR / "best_model.pt"
VOCAB_PATH      = ARTIFACTS_DIR / "vocab.json"
HYPERPARAMS_PATH = ARTIFACTS_DIR / "hyperparams.json"

DEFAULT_HYPERPARAMS = {
    "emsize": 200,
    "d_hid": 200,
    "nlayers": 2,
    "nhead": 2,
    "dropout": 0.3,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts from %s ...", ARTIFACTS_DIR)

    # Vocabulary
    if not VOCAB_PATH.exists():
        raise RuntimeError(f"vocab.json not found: {VOCAB_PATH}. Run scripts/export_artifacts.py first.")
    state.vocab = Vocab.load(str(VOCAB_PATH))
    ntokens = len(state.vocab)
    logger.info("Vocabulary: %d tokens", ntokens)

    # Hyperparameters
    if HYPERPARAMS_PATH.exists():
        hp = json.loads(HYPERPARAMS_PATH.read_text())
        logger.info("Hyperparameters loaded from %s", HYPERPARAMS_PATH)
    else:
        hp = DEFAULT_HYPERPARAMS
        logger.warning("hyperparams.json missing — using defaults.")

    # Model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"best_model.pt not found: {MODEL_PATH}. Run scripts/export_artifacts.py first.")

    state.model = TransformerModel(
        ntoken=ntokens,
        d_model=hp.get("emsize",  DEFAULT_HYPERPARAMS["emsize"]),
        nhead=hp.get("nhead",     DEFAULT_HYPERPARAMS["nhead"]),
        d_hid=hp.get("d_hid",    DEFAULT_HYPERPARAMS["d_hid"]),
        nlayers=hp.get("nlayers", DEFAULT_HYPERPARAMS["nlayers"]),
        dropout=hp.get("dropout", DEFAULT_HYPERPARAMS["dropout"]),
    ).to(state.device)

    state.model.load_state_dict(torch.load(str(MODEL_PATH), map_location=state.device))
    state.model.eval()

    total_params = sum(p.numel() for p in state.model.parameters())
    logger.info("Model ready — %d parameters", total_params)

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Shakespeare Text Generator",
        description="Transformer language model trained on Tiny Shakespeare.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
