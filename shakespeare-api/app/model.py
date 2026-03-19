"""
model.py — Transformer language model for Shakespeare text generation.

Classes and functions match the notebook exactly so that saved weights load correctly.
"""

import json
import math
import re
from collections import Counter
from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def basic_english_tokenize(text: str) -> List[str]:
    """
    Lowercases text, pads punctuation with spaces, splits on whitespace.
    Must match the notebook tokenizer exactly.
    """
    text = text.lower()
    text = re.sub(r'([.,!?;:\'\"\(\)\[\]\-])', r' \1 ', text)
    tokens = text.split()
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """
    Minimal vocabulary: maps tokens <-> integer indices, sorted by frequency.
    Special tokens (e.g. <unk>) are placed first.
    Matches the notebook Vocab class exactly.
    """

    def __init__(self, token_counts: Counter, specials: Optional[List[str]] = None):
        self.itos: List[str] = []
        self.stoi: dict = {}
        self.default_index: int = 0

        if specials:
            for s in specials:
                self.stoi[s] = len(self.itos)
                self.itos.append(s)

        for token, _ in token_counts.most_common():
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def get_itos(self) -> List[str]:
        return self.itos

    # ------------------------------------------------------------------
    # Serialization helpers (not in original notebook — added for the API)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize vocabulary to a plain dict for JSON storage."""
        return {
            "itos": self.itos,
            "stoi": self.stoi,
            "default_index": self.default_index,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Vocab":
        """Reconstruct a Vocab from the dict produced by to_json()."""
        # Build with empty counter so no tokens are auto-added
        vocab = cls.__new__(cls)
        vocab.itos = data["itos"]
        vocab.stoi = data["stoi"]
        vocab.default_index = data["default_index"]
        return vocab

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load a Vocab from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)

    def save(self, path: str) -> None:
        """Save this Vocab to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Injects sine/cosine positional information into token embeddings.
    Matches the notebook implementation exactly.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    Embedding → PositionalEncoding → TransformerEncoder → Linear output.
    Architecture matches the notebook exactly so that saved weights load.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Upper-triangular -inf matrix for causal (autoregressive) masking."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(
    model: nn.Module,
    vocab: Vocab,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a text continuation from a prompt.

    Args:
        model:          Trained TransformerModel (in eval mode).
        vocab:          Vocab built from the training data.
        prompt:         Seed text, e.g. "to be or not to be".
        max_new_tokens: Number of new tokens to generate.
        temperature:    <1 = conservative, 1 = normal, >1 = creative.
        top_k:          If >0, restrict sampling to top-k candidates.
        device:         Torch device to run inference on.

    Returns:
        Full generated string (prompt tokens + new tokens joined by spaces).
    """
    model.eval()

    tokens = basic_english_tokenize(prompt)
    token_indices = vocab(tokens)

    if not token_indices:
        return prompt

    input_ids = torch.tensor(token_indices, dtype=torch.long).unsqueeze(1).to(device)
    generated_tokens = list(token_indices)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(0)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

            output = model(input_ids, src_mask)
            logits = output[-1, 0, :]  # logits for last position

            logits = logits / temperature

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                threshold = top_k_values[-1]
                logits[logits < threshold] = float("-inf")

            probs = torch.softmax(logits, dim=0)

            if top_k == 0 and temperature == 1.0:
                next_token = torch.argmax(probs).unsqueeze(0)
            else:
                next_token = torch.multinomial(probs, 1)

            generated_tokens.append(next_token.item())

            next_token = next_token.unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=0)

            # Cap context window to 256 tokens
            if input_ids.size(0) > 256:
                input_ids = input_ids[-256:]

    itos = vocab.get_itos()
    words = [itos[idx] for idx in generated_tokens]
    return " ".join(words)
