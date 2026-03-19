# Shakespeare Text Generator — REST API

A FastAPI microservice wrapping a PyTorch Transformer language model trained on
the Tiny Shakespeare dataset. Deployed to Google Cloud Run.

---

## Project structure

```
shakespeare-api/
├── app.py            # FastAPI app — routes, startup, request/response schemas
├── model.py          # TransformerModel, Vocab, tokenizer, generate()
├── requirements.txt  # Pinned dependencies (CPU-only torch)
├── Dockerfile        # Container for Cloud Run
├── best_model.pt     # ← YOU ADD THIS (exported from Assignment 4 notebook)
├── vocab.json        # ← YOU ADD THIS (exported from Assignment 4 notebook)
└── hyperparams.json  # ← YOU ADD THIS (optional but recommended)
```

---

## Step 1 — Export model artifacts from your notebook

Add a new cell at the end of your Assignment 4 notebook and run it:

```python
import json, torch

# 1. Save model weights (must still have `model` in scope)
torch.save(model.state_dict(), "best_model.pt")
print("Saved best_model.pt")

# 2. Save vocabulary
vocab_data = {
    "itos": vocab.get_itos(),
    "stoi": vocab.stoi,
    "default_index": vocab.default_index,
}
with open("vocab.json", "w") as f:
    json.dump(vocab_data, f)
print(f"Saved vocab.json ({len(vocab)} tokens)")

# 3. Save hyperparameters
hp = {
    "emsize": emsize,     # 200
    "d_hid": d_hid,       # 200
    "nlayers": nlayers,   # 2
    "nhead": nhead,        # 2
    "dropout": dropout,   # 0.3
}
with open("hyperparams.json", "w") as f:
    json.dump(hp, f)
print("Saved hyperparams.json")
```

Move the three generated files into `shakespeare-api/`:

```bash
mv best_model.pt vocab.json hyperparams.json shakespeare-api/
```

---

## Step 2 — Run locally

```bash
cd shakespeare-api

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --port 8080
```

The API will be available at `http://localhost:8080`.
Interactive docs: `http://localhost:8080/docs`

---

## Step 3 — Test locally with curl

```bash
# Health check
curl http://localhost:8080/health

# Service info
curl http://localhost:8080/

# Generate text (balanced mode)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "to be or not to be", "max_tokens": 50, "temperature": 0.8, "top_k": 10}'

# Generate text (greedy)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "the king shall", "max_tokens": 40, "temperature": 1.0, "top_k": 0}'

# Generate text (creative)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "romeo , romeo , wherefore art thou", "max_tokens": 60, "temperature": 1.3, "top_k": 20}'
```

---

## Step 4 — Deploy to Google Cloud Run

### Prerequisites

```bash
# Authenticate
gcloud auth login

# Set your project (replace YOUR_PROJECT_ID)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs (one-time)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Deploy

Run this from inside the `shakespeare-api/` directory:

```bash
gcloud run deploy shakespeare-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 60 \
  --platform managed
```

Cloud Build will build the Docker image, push it to Artifact Registry, and deploy
to Cloud Run. The command prints the live URL when it finishes.

### Test the live deployment

Replace `YOUR_URL` with the URL printed by the deploy command:

```bash
export BASE_URL="https://YOUR_URL"

curl $BASE_URL/health

curl -X POST $BASE_URL/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "to be or not to be", "max_tokens": 50, "temperature": 0.8, "top_k": 10}'
```

### View logs

```bash
gcloud run logs read shakespeare-api --region us-central1
```

---

## API reference

### `GET /`
Returns service metadata.

### `GET /health`
Returns `{"status": "healthy"}`. Used by Cloud Run health checks.

### `POST /generate`

Request body:

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `prompt` | string | required | ≥1 char | Seed text |
| `max_tokens` | int | 50 | 1–200 | Tokens to generate |
| `temperature` | float | 0.8 | 0.1–2.0 | Sampling temperature |
| `top_k` | int | 10 | 0–50 | Top-k sampling (0 = greedy) |

Response:

```json
{
  "prompt": "to be or not to be",
  "generated_text": "to be or not to be , that is the question ...",
  "parameters": {"max_tokens": 50, "temperature": 0.8, "top_k": 10},
  "tokens_generated": 50
}
```

---

## Notes

- PyTorch runs on **CPU only** — no GPU needed on Cloud Run.
- The model loads once at startup; all requests share the same loaded model.
- Cloud Run scales to zero when idle, so the first request after a cold start
  may take ~5–10 seconds as PyTorch initializes.
- Memory: 1Gi is the minimum recommended (PyTorch + weights ≈ 500 MB).
