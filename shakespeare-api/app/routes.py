import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.model import basic_english_tokenize, generate
from app.schemas import GenerateRequest, GenerateResponse
from app.state import state

logger = logging.getLogger(__name__)

router = APIRouter()

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shakespeare Text Generator</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      background: #0f0e17;
      color: #fffffe;
      font-family: 'Georgia', serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2.5rem 1rem 4rem;
    }

    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    header h1 {
      font-size: 2.2rem;
      letter-spacing: 0.04em;
      color: #ff8906;
    }
    header p {
      margin-top: 0.5rem;
      color: #a7a9be;
      font-size: 0.95rem;
    }

    .card {
      background: #1a1a2e;
      border: 1px solid #2e2e4a;
      border-radius: 12px;
      padding: 2rem;
      width: 100%;
      max-width: 680px;
    }

    label {
      display: block;
      font-size: 0.85rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #a7a9be;
      margin-bottom: 0.4rem;
    }

    textarea, input[type="number"] {
      width: 100%;
      background: #0f0e17;
      border: 1px solid #2e2e4a;
      border-radius: 8px;
      color: #fffffe;
      font-family: 'Georgia', serif;
      font-size: 1rem;
      padding: 0.7rem 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea { resize: vertical; min-height: 90px; }
    textarea:focus, input[type="number"]:focus { border-color: #ff8906; }

    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1.2rem; }

    .field { display: flex; flex-direction: column; }
    .hint  { font-size: 0.72rem; color: #6e6e8a; margin-top: 0.25rem; }

    .modes {
      display: flex;
      gap: 0.6rem;
      margin-top: 1.2rem;
      flex-wrap: wrap;
    }
    .mode-btn {
      flex: 1;
      padding: 0.5rem 0.7rem;
      border: 1px solid #2e2e4a;
      border-radius: 8px;
      background: #0f0e17;
      color: #a7a9be;
      cursor: pointer;
      font-size: 0.82rem;
      text-align: center;
      transition: border-color 0.2s, color 0.2s;
    }
    .mode-btn:hover { border-color: #ff8906; color: #fffffe; }
    .mode-btn.active { border-color: #ff8906; color: #ff8906; }

    button#generateBtn {
      margin-top: 1.6rem;
      width: 100%;
      padding: 0.85rem;
      background: #ff8906;
      color: #0f0e17;
      border: none;
      border-radius: 8px;
      font-family: 'Georgia', serif;
      font-size: 1rem;
      font-weight: bold;
      cursor: pointer;
      letter-spacing: 0.04em;
      transition: background 0.2s, opacity 0.2s;
    }
    button#generateBtn:disabled { opacity: 0.5; cursor: not-allowed; }
    button#generateBtn:hover:not(:disabled) { background: #e07800; }

    #result {
      margin-top: 2rem;
      width: 100%;
      max-width: 680px;
      display: none;
    }
    #result h2 {
      font-size: 0.85rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #a7a9be;
      margin-bottom: 0.6rem;
    }
    #outputBox {
      background: #1a1a2e;
      border: 1px solid #2e2e4a;
      border-radius: 12px;
      padding: 1.4rem;
      line-height: 1.8;
      font-size: 1.05rem;
      color: #fffffe;
      white-space: pre-wrap;
    }
    #outputBox .prompt-part { color: #ff8906; }

    .meta {
      margin-top: 0.6rem;
      font-size: 0.78rem;
      color: #6e6e8a;
    }

    #error {
      margin-top: 1rem;
      padding: 0.8rem 1rem;
      background: #2e0a0a;
      border: 1px solid #8b0000;
      border-radius: 8px;
      color: #ff6b6b;
      font-size: 0.9rem;
      display: none;
    }

    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid #0f0e17;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <h1>&#x2767; Shakespeare Text Generator</h1>
    <p>Transformer language model trained on Tiny Shakespeare &mdash; deployed on Google Cloud Run</p>
  </header>

  <div class="card">
    <label for="prompt">Prompt</label>
    <textarea id="prompt" placeholder="to be or not to be">to be or not to be</textarea>

    <div class="modes">
      <div class="mode-btn" data-temp="1.0" data-topk="0">Greedy</div>
      <div class="mode-btn active" data-temp="0.8" data-topk="10">Balanced</div>
      <div class="mode-btn" data-temp="1.3" data-topk="20">Creative</div>
    </div>

    <div class="grid">
      <div class="field">
        <label for="maxTokens">Max Tokens</label>
        <input type="number" id="maxTokens" value="50" min="1" max="200" />
        <span class="hint">1 – 200</span>
      </div>
      <div class="field">
        <label for="temperature">Temperature</label>
        <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.1" />
        <span class="hint">0.1 – 2.0</span>
      </div>
      <div class="field">
        <label for="topK">Top-K</label>
        <input type="number" id="topK" value="10" min="0" max="50" />
        <span class="hint">0 = greedy</span>
      </div>
    </div>

    <button id="generateBtn">Generate</button>
  </div>

  <div id="error"></div>

  <div id="result">
    <h2>Generated Text</h2>
    <div id="outputBox"></div>
    <div class="meta" id="meta"></div>
  </div>

  <script>
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('temperature').value = btn.dataset.temp;
        document.getElementById('topK').value = btn.dataset.topk;
      });
    });

    document.getElementById('generateBtn').addEventListener('click', async () => {
      const prompt      = document.getElementById('prompt').value.trim();
      const max_tokens  = parseInt(document.getElementById('maxTokens').value);
      const temperature = parseFloat(document.getElementById('temperature').value);
      const top_k       = parseInt(document.getElementById('topK').value);

      const errBox = document.getElementById('error');
      errBox.style.display = 'none';

      if (!prompt) { errBox.textContent = 'Please enter a prompt.'; errBox.style.display = 'block'; return; }

      const btn = document.getElementById('generateBtn');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Generating…';

      try {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, max_tokens, temperature, top_k }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || res.statusText);
        }
        const data = await res.json();

        const box = document.getElementById('outputBox');
        const generated = data.generated_text.slice(prompt.length);
        box.innerHTML = '<span class="prompt-part">' + escHtml(prompt) + '</span>' + escHtml(generated);

        document.getElementById('meta').textContent =
          `${data.tokens_generated} tokens generated  ·  temperature ${temperature}  ·  top-k ${top_k}`;

        document.getElementById('result').style.display = 'block';
      } catch (e) {
        errBox.textContent = 'Error: ' + e.message;
        errBox.style.display = 'block';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
      }
    });

    function escHtml(str) {
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
  </script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse(content=_UI_HTML)


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
