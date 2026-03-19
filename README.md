# Shakespeare Text Generator — REST API

A FastAPI microservice that serves a PyTorch Transformer language model trained on the
Tiny Shakespeare dataset (~1.1 MB of Shakespeare's plays). Supports greedy, balanced,
and creative text generation via a simple REST API. Deployed on Google Cloud Run.

---

## Project structure

```
shakespeare-api/
├── app/
│   ├── main.py       # FastAPI factory, startup, CORS
│   ├── routes.py     # GET /, GET /health, POST /generate
│   ├── schemas.py    # Request / response Pydantic models
│   ├── model.py      # TransformerModel, Vocab, generate()
│   └── state.py      # Loaded model + vocab singleton
├── artifacts/        # Model weights & vocab (not committed to git)
│   ├── best_model.pt
│   ├── vocab.json
│   └── hyperparams.json
├── scripts/
│   └── export_artifacts.py   # Re-trains and saves artifacts
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.11+
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (for deployment only)

---

## Step 1 — Generate model artifacts

The model weights and vocabulary are not committed to git. Run the export script to
train the model and save the artifacts:

```bash
cd shakespeare-api
python3 -m venv .venv
source .venv/bin/activate
pip install torch requests       # minimal deps for training
python scripts/export_artifacts.py
```

This downloads Tiny Shakespeare, trains for 10 epochs (~2.5 min on Apple Silicon),
and writes three files to `artifacts/`:

| File | Size | Description |
|---|---|---|
| `best_model.pt` | ~22 MB | Trained weights (best validation loss) |
| `vocab.json` | ~293 KB | 10,926-token vocabulary |
| `hyperparams.json` | <1 KB | Model architecture config |

---

## Step 2 — Install dependencies & run locally

```bash
cd shakespeare-api
source .venv/bin/activate        # if not already active
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

The server starts at **http://localhost:8080**.
Interactive API docs: **http://localhost:8080/docs**

You should see this in the terminal on successful startup:

```
INFO: Vocabulary: 10926 tokens
INFO: Model ready — 4865326 parameters
INFO: Uvicorn running on http://127.0.0.1:8080
```

---

## Step 3 — Test with curl

**Health check**
```bash
curl http://localhost:8080/health
```
```json
{"status": "healthy"}
```

**Service info**
```bash
curl http://localhost:8080/
```

**Generate — balanced** `(temperature=0.8, top_k=10)` — recommended default
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "to be or not to be", "max_tokens": 50, "temperature": 0.8, "top_k": 10}'
```

**Generate — greedy** `(temperature=1.0, top_k=0)` — deterministic, same output every time
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "the king shall", "max_tokens": 40, "temperature": 1.0, "top_k": 0}'
```

**Generate — creative** `(temperature=1.3, top_k=20)` — more varied output
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "romeo , romeo , wherefore art thou", "max_tokens": 60, "temperature": 1.3, "top_k": 20}'
```

**Example response**
```json
{
  "prompt": "to be or not to be",
  "generated_text": "to be or not to be the king richard iii : and that is it ...",
  "parameters": {
    "max_tokens": 50,
    "temperature": 0.8,
    "top_k": 10
  },
  "tokens_generated": 50
}
```

---

## Step 4 — Deploy to Google Cloud Run

### One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Deploy

Run from inside the `shakespeare-api/` directory:

```bash
gcloud run deploy shakespeare-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 60 \
  --platform managed
```

Cloud Build builds the Docker image and deploys it. The live URL is printed when
the command completes.

### Test the live service

```bash
export BASE_URL="https://YOUR-CLOUD-RUN-URL"

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
Returns service metadata and status.

### `GET /health`
Returns `{"status": "healthy"}`. Used by Cloud Run for health checks.

### `POST /generate`

**Request body**

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `prompt` | string | required | ≥ 1 char | Seed text for generation |
| `max_tokens` | int | `50` | 1 – 200 | Number of tokens to generate |
| `temperature` | float | `0.8` | 0.1 – 2.0 | Lower = conservative, higher = creative |
| `top_k` | int | `10` | 0 – 50 | Candidate pool size (`0` = greedy) |

**Generation modes**

| Mode | temperature | top_k | Behaviour |
|---|---|---|---|
| Greedy | `1.0` | `0` | Always picks highest-probability token — deterministic but repetitive |
| Balanced | `0.8` | `10` | Samples from top 10 — coherent and varied |
| Creative | `1.3` | `20` | Wider sampling — more surprising combinations |

---

## Notes

- **CPU only** — PyTorch runs on CPU inside the container. No GPU is needed on Cloud Run.
- **Cold start** — Cloud Run scales to zero when idle. The first request after inactivity
  takes ~5–10 seconds while the model loads (~22 MB weights + PyTorch init).
- **Memory** — 1Gi is the minimum; PyTorch + model weights use ~500 MB at runtime.
- **Greedy repetition** — Greedy decoding is known to produce loops ("and i have heard...
  and i have heard..."). Use balanced or creative mode for better-looking output.
