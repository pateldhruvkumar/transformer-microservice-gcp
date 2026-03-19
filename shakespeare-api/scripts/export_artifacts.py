"""
scripts/export_artifacts.py

Replicates the Assignment 4 notebook exactly, then saves three files to
shakespeare-api/artifacts/:
  - best_model.pt      (model weights)
  - vocab.json         (vocabulary)
  - hyperparams.json   (model architecture config)

Run from the shakespeare-api/ directory:
    python scripts/export_artifacts.py
"""

from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

import json
import math
import os
import random
import re
import time
from collections import Counter
from typing import List, Tuple

import requests
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Use MPS on Apple Silicon if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Download Tiny Shakespeare
# ---------------------------------------------------------------------------
print("\n[1/7] Downloading Tiny Shakespeare...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
raw_text = response.text
print(f"  Total characters: {len(raw_text):,}")

# ---------------------------------------------------------------------------
# 2. Train/Val/Test split (identical to notebook)
# ---------------------------------------------------------------------------
print("[2/7] Splitting data...")
random.seed(42)
lines = raw_text.splitlines()
lines = [line for line in lines if line.strip()]
random.shuffle(lines)

total_lines = len(lines)
train_end = int(total_lines * 0.90)
val_end   = int(total_lines * 0.95)

train_text = lines[:train_end]
val_text   = lines[train_end:val_end]
test_text  = lines[val_end:]
print(f"  Train: {len(train_text):,}  Val: {len(val_text):,}  Test: {len(test_text):,}")

# ---------------------------------------------------------------------------
# 3. Tokenizer & Vocabulary (identical to notebook)
# ---------------------------------------------------------------------------
print("[3/7] Building vocabulary...")

def basic_english_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'([.,!?;:\'\"\(\)\[\]\-])', r' \1 ', text)
    return text.split()


class Vocab:
    def __init__(self, token_counts: Counter, specials: List[str] = None):
        self.itos = []
        self.stoi = {}
        self.default_index = 0
        if specials:
            for s in specials:
                self.stoi[s] = len(self.itos)
                self.itos.append(s)
        for token, _ in token_counts.most_common():
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def set_default_index(self, index: int):
        self.default_index = index

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def get_itos(self) -> List[str]:
        return self.itos


counter = Counter()
for tokens in map(basic_english_tokenize, train_text):
    counter.update(tokens)
vocab = Vocab(counter, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print(f"  Vocabulary size: {len(vocab):,}")

# ---------------------------------------------------------------------------
# 4. Process data into tensors (identical to notebook)
# ---------------------------------------------------------------------------
print("[4/7] Tokenising and batchifying...")

def data_process(raw_text_iter) -> Tensor:
    data = [torch.tensor(vocab(basic_english_tokenize(item)), dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int) -> Tensor:
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size      = 20
eval_batch_size = 10

train_data = batchify(data_process(train_text), batch_size)
val_data   = batchify(data_process(val_text),   eval_batch_size)
test_data  = batchify(data_process(test_text),  eval_batch_size)
print(f"  train_data: {train_data.shape}  val_data: {val_data.shape}")

# ---------------------------------------------------------------------------
# 5. Model classes (identical to notebook)
# ---------------------------------------------------------------------------
print("[5/7] Building model...")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.linear(output)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


# Hyperparameters (identical to notebook)
ntokens = len(vocab)
emsize  = 200
d_hid   = 200
nlayers = 2
nhead   = 2
dropout = 0.3
bptt    = 35

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# ---------------------------------------------------------------------------
# 6. Training (identical to notebook)
# ---------------------------------------------------------------------------
print("[6/7] Training (10 epochs)...")

criterion = nn.CrossEntropyLoss()
lr        = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data    = source[i : i + seq_len]
    target  = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def train_epoch(model: nn.Module) -> None:
    model.train()
    total_loss = 0.0
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

        output = model(data, src_mask)
        loss   = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            ms_per   = (time.time() - start_time) * 1000 / log_interval
            print(f"  | epoch {epoch:2d} | {batch:4d}/{num_batches:4d} batches "
                  f"| lr {scheduler.get_last_lr()[0]:.2f} "
                  f"| ms/batch {ms_per:.1f} | loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.1f}")
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = generate_square_subsequent_mask(seq_len).to(device)
            output = model(data, src_mask)
            total_loss += seq_len * criterion(output.view(-1, ntokens), targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float("inf")
best_state    = None
epochs        = 10
total_start   = time.time()

for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_epoch(model)
    val_loss  = evaluate(model, val_data)
    val_ppl   = math.exp(val_loss)
    train_loss = evaluate(model, train_data)
    elapsed   = time.time() - t0

    print("-" * 72)
    print(f"  epoch {epoch:2d} | {elapsed:.1f}s "
          f"| train loss {train_loss:.2f} (ppl {math.exp(train_loss):.1f}) "
          f"| val loss {val_loss:.2f} (ppl {val_ppl:.1f})")
    print("-" * 72)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ✓ New best model (val loss {val_loss:.3f})")

    scheduler.step()

model.load_state_dict(best_state)
print(f"\n  Total training time: {time.time() - total_start:.1f}s")
print(f"  Best val loss: {best_val_loss:.2f} (ppl {math.exp(best_val_loss):.1f})")

# ---------------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------------
print("\n[7/7] Saving artifacts...")

# Model weights — save to CPU so they load on any device
cpu_state = {k: v.cpu() for k, v in best_state.items()}
torch.save(cpu_state, ARTIFACTS_DIR / "best_model.pt")
print("  Saved artifacts/best_model.pt")

# Vocabulary
with open(ARTIFACTS_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump({"itos": vocab.get_itos(), "stoi": vocab.stoi,
               "default_index": vocab.default_index}, f, ensure_ascii=False)
print(f"  Saved artifacts/vocab.json  ({len(vocab):,} tokens)")

# Hyperparameters
with open(ARTIFACTS_DIR / "hyperparams.json", "w") as f:
    json.dump({"emsize": emsize, "d_hid": d_hid, "nlayers": nlayers,
               "nhead": nhead, "dropout": dropout}, f)
print("  Saved artifacts/hyperparams.json")

print(f"\nDone. Artifacts written to {ARTIFACTS_DIR.resolve()}")
