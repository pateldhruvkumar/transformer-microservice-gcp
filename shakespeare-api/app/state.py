"""
Global application state — holds the loaded model and vocabulary.

Populated once during FastAPI lifespan startup; shared across all requests.
"""

from __future__ import annotations

from typing import Optional

import torch

from app.model import TransformerModel, Vocab


class AppState:
    model: Optional[TransformerModel] = None
    vocab: Optional[Vocab] = None
    device: torch.device = torch.device("cpu")

    @property
    def ready(self) -> bool:
        return self.model is not None and self.vocab is not None


# Module-level singleton imported by both main.py and routes.py
state = AppState()
