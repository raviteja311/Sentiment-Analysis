# Sentiment Analysis

Three-class sentiment classification for tweets - **negative**, **neutral**, **positive** -
with four models trained on [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
(`sentiment` config). Served as a FastAPI service and a Streamlit app, with a
containerised CPU image and CI.

## Results

Measured on the full test split (12,284 tweets). Every figure here is generated
by `make evaluate` and read from `reports/metrics/*.json` - **do not edit this
table by hand**, regenerate it.

| Model | Split | Examples | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|---|---|
| Twitter-RoBERTa | validation | 2000 | 0.7945 | 0.7864 | 0.7539 | 0.7637 | 0.8417 |
| Twitter-RoBERTa | test | 12284 | 0.7109 | 0.7124 | 0.7328 | 0.6954 | 0.7091 |
| Bi-GRU | validation | 2000 | 0.6800 | 0.6620 | 0.5882 | 0.6937 | 0.7041 |
| Bi-GRU | test | 12284 | 0.6343 | 0.6223 | 0.6317 | 0.6549 | 0.5803 |
| Bi-LSTM | validation | 2000 | 0.6715 | 0.6470 | 0.5612 | 0.6934 | 0.6864 |
| Bi-LSTM | test | 12284 | 0.6370 | 0.6181 | 0.6063 | 0.6757 | 0.5722 |
| Logistic Regression | validation | 2000 | 0.6550 | 0.6349 | 0.5344 | 0.6496 | 0.7208 |
| Logistic Regression | test | 12284 | 0.5806 | 0.5768 | 0.5990 | 0.5774 | 0.5540 |

The transformer leads by a clear margin. The two recurrent models are within
half a point of each other - treat them as equivalent - and beat the linear
baseline by 5-6 points of accuracy. Both initialise their embeddings from
GloVe Twitter vectors (95.5% vocabulary coverage), which is worth roughly
+0.03 macro F1 and +0.05 positive-class F1 over learning them from scratch.
See [MODEL_CARD.md](MODEL_CARD.md) for limitations and intended use.

Each record in `reports/metrics/` stores the hyperparameters, the split sizes and
the class distribution alongside the scores, so any number above can be traced
to what produced it.

## Models

| Key | Model | Framework | Artifact |
|---|---|---|---|
| `lr` | TF-IDF + Logistic Regression | scikit-learn | `models/lr/pipeline.joblib` |
| `lstm` | Bi-LSTM over GloVe Twitter embeddings | Keras 3 | `models/lstm/model_final.keras` |
| `gru` | Bi-GRU over GloVe Twitter embeddings | Keras 3 | `models/gru/model_final.keras` |
| `roberta` | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment` | Transformers | `models/roberta/` |

The transformer was keyed as `bert` until the name was corrected to match the
checkpoint - it is **RoBERTa** (`model_type: roberta`, byte-pair tokenizer).
`"model": "bert"` is still accepted as a deprecated alias and resolves to
`roberta`, so existing requests keep working.

## Dataset

| Split | Examples |
|---|---|
| train | 45,615 |
| validation | 2,000 |
| test | 12,284 |

The training split is imbalanced: neutral 20,673, positive 17,849, negative
7,093. The dataset downloads automatically on the first training or evaluation
run; it is not needed for inference.

## Quick start

Requires **Python 3.12** (3.11 is also covered by CI).

```bash
git clone https://github.com/raviteja311/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements/inference-cpu.txt            # or: make install
make fetch-weights                               # 574 MB, see Weights below
```

Keep the `--extra-index-url`. Without it pip downloads the default PyTorch
wheel, which bundles CUDA and pulls in several gigabytes of NVIDIA libraries
that a CPU install never uses. On macOS and Linux ARM the requirements file
selects the plain `tensorflow` package automatically, because `tensorflow-cpu`
has no wheels for those platforms.

### Weights

The large weights are **not in git**. They live on the Hugging Face Hub at
[RAVITEJA311/sentiment-analysis-models](https://huggingface.co/RAVITEJA311/sentiment-analysis-models)
and are fetched on demand:

```bash
make fetch-weights                        # all of them, 574 MB
python -m src.artifacts --models lstm     # or just one
```

Point it elsewhere with `SENTIMENT_MODELS_REPO`.

Downloads are pinned and verified. `models/remote_manifest.json` records the
Hub commit that matches the committed tokenizers and calibration files, and the
sha256 of every fetched file; a download that does not match is deleted and
reported. `SENTIMENT_MODELS_REVISION` overrides the pin. After publishing new
weights, `make publish-weights` rewrites the manifest so the pin moves with
them; `python -m src.artifacts --write-manifest` does the same for weights
already on the Hub.

Skipping this step is not fatal. The Logistic Regression pipeline, the
tokenizers and the transformer's config and vocabulary are small enough to stay
in git, so a plain clone still serves predictions from `lr`; the neural models
report themselves unavailable with the command that fixes it, the UI hides them,
and the API answers 503 rather than 500.

### API

```bash
make api          # or: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; answers even when no model is usable |
| `GET /ready` | readiness; 503 when no model can be served |
| `GET /models` | every model: availability, reason, version and per-file digests |
| `POST /predict` | classify one text |
| `POST /predict/batch` | classify up to 256 texts |

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"this is fantastic","model":"roberta"}'
```

```json
{
  "model": "roberta",
  "version": "a378eddd7113",
  "label": "positive",
  "confidence": 0.9474,
  "probabilities": {"negative": 0.0210, "neutral": 0.0315, "positive": 0.9474}
}
```

`version` is a digest of the artifact that produced the prediction - the
weights, the tokenizer and the calibration file - so a logged prediction stays
traceable after a retrain or a re-fetch.

`GET /models` reports the version **actually being served**. A predictor is
cached once loaded, so replacing weights on disk does not change what answers
requests; the response therefore also carries `loaded`, `version_on_disk` and
`stale`, which tells you a restart is needed rather than quietly reporting a
version nothing is using.

`confidence` is **calibrated**, not a raw softmax - see
[Calibration](#calibration).

Interactive docs are at `/docs`.

Models load on first use, so the process is up quickly and a broken artifact
cannot stop it from starting. Three optional variables tune serving:

| Variable | Default | Purpose |
|---|---|---|
| `PRELOAD_MODELS` | unset | keys to load at startup, e.g. `lr,roberta` or `all`; one that fails to load is logged and skipped |
| `MAX_CONCURRENT_INFERENCE` | unset (unlimited) | cap on simultaneous forward passes; the endpoints run in a 40-thread pool, which oversubscribes a CPU |
| `TORCH_NUM_THREADS` | torch's default | intra-op threads for the transformer, applied once at load |

### Logging

The service logs one JSON object per line. Every request is assigned an
`X-Request-ID` (an inbound one is honoured if it is safe to log) which is
echoed in the response and attached to every log line produced while serving it,
so a prediction can be traced from the access line to the model version that
produced it:

```json
{"timestamp": "2026-09-22T10:17:20.105Z", "level": "INFO", "logger": "api.access",
 "message": "request", "request_id": "ad56b98b50aa", "method": "POST",
 "path": "/predict", "status": 200, "duration_ms": 1.59}
```

**The text being classified is never logged** - only its length. Set
`LOG_FORMAT=plain` for readable local output and `LOG_LEVEL` to change verbosity.

### Rate limiting

Callers get `60/minute` by default, **shared across endpoints** so the budget
cannot be spent twice by alternating between `/predict` and `/predict/batch`.
Exceeding it returns **429** with a `Retry-After` header, and the remaining
budget is reported in `X-RateLimit-*`. Health, readiness, `/models` and the
OpenAPI docs are exempt - throttling a liveness probe gets a healthy container
restarted, and throttling `/docs` just makes the service look broken.

That limit counts requests, and a batch is one request however many texts it
carries. A second budget therefore counts **texts**: `/predict` costs 1,
`/predict/batch` costs one per text, and a batch that would overrun the budget
is rejected whole with **429** without consuming anything. A batch larger than
the entire budget says so in its `detail`, since waiting would not help.

| Variable | Default | Purpose |
|---|---|---|
| `RATE_LIMIT` | `60/minute` | request limit, or `off` to disable |
| `TEXT_RATE_LIMIT` | `1024/minute` | text limit (a batch of N costs N), or `off` to disable |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | set to `redis://...` to share counters across workers |
| `TRUST_PROXY_HEADERS` | unset | use `X-Forwarded-For` for the caller's identity |

Two limits of this, worth knowing before relying on it. With the in-memory
store each worker counts separately, so N workers give a caller N times the
limit; use Redis to share state. And `TRUST_PROXY_HEADERS` should only be set
when a proxy you control overwrites that header - otherwise a caller sets it
themselves and bypasses the limit.

**This is a backstop, not the real defence.** Rate limiting belongs at the edge,
in an ingress or API gateway, where it applies before a request reaches any
application process.

### Streamlit app

```bash
make ui           # or: streamlit run app/streamlit_app.py
```

### Docker

```bash
docker build --target cpu -t sentiment-analysis-api:local .
docker run --rm -p 8000:8000 sentiment-analysis-api:local
docker compose up --build      # API on 8000, UI on 8501
```

The image carries only the small artifacts, so it serves `lr` out of the box and
stays under a gigabyte. Bake the rest in at build time, or mount them at run
time as docker-compose does:

```bash
docker build --target cpu --build-arg FETCH_MODELS="lstm gru roberta" .
```

## Calibration

Raw softmax outputs are overconfident: before calibration the transformer
averaged 0.915 confidence while being right 71.1% of the time. `make calibrate`
fits a temperature per model on the validation split, adopts it only if it also
helps on held-out validation folds, and stores it in
`models/<key>/calibration.json`; the predictor applies it, so the API, the UI
and the evaluation all report the same calibrated numbers. This table is
printed by `python -m src.calibration --table-only` - regenerate it, do not
edit it.

| Model | Temperature | ECE before | ECE after | Mean confidence before | after | Accuracy |
|---|---|---|---|---|---|---|
| roberta | 1.99 | 0.2039 | 0.0865 | 0.915 | 0.796 | 0.7109 |
| lstm | 1.12 | 0.0482 | 0.0248 | 0.684 | 0.659 | 0.6370 |
| gru | 1.18 | 0.0510 | 0.0226 | 0.683 | 0.649 | 0.6343 |
| lr | 1.00 (not adopted) | 0.0444 | 0.0444 | 0.611 | 0.611 | 0.5806 |

Fitted on validation, measured on test. Temperature scaling is monotonic, so no
prediction changes and the results table above is unaffected - only the spread
of the probabilities moves.

The linear baseline is left uncalibrated. The temperature is constrained to at
least 1.0, so calibration can only soften confidence, and the bounded fit lands
on exactly 1.0. Unconstrained, it fits 0.94 on validation and gets *worse* on
test (ECE 0.0444 -> 0.0517): the model is underconfident on validation (63.3%
confidence, 65.5% accuracy) and overconfident on test (61.1% confidence, 58.1%
accuracy), because validation is the easier split. It is therefore left alone
rather than sharpened.

## Retraining

```bash
pip install -r requirements/train.txt
make train-lr        # ~5 min, CPU
make train-lstm      # ~6 min, CPU
make train-gru       # ~6 min, CPU
make train-roberta   # ~2h15m on a GTX 1650 with mixed precision
make evaluate        # scores every available model, regenerates the table above
make calibrate       # refits temperature scaling after retraining
```

Training writes both the artifact and a metrics record to `reports/metrics/`.
`make evaluate` merges its results into those records rather than replacing
them, so the training provenance survives. Publish retrained weights with
`make publish-weights` (needs `huggingface-cli login`).

TensorFlow has no native Windows GPU support from 2.11 onward, so the LSTM and
GRU train on CPU there regardless of what hardware is present. The transformer
uses CUDA when available and enables mixed precision automatically, which is
what keeps it inside 4 GB of VRAM.

## Tests and quality

```bash
make test         # 297 tests
make lint         # ruff + black
```

Tests skip themselves rather than fail when a model's weights are absent, so the
suite is meaningful on a clone where `make fetch-weights` has not been run. CI runs lint, the test suite on
Python 3.11 and 3.12, and a container smoke test that asserts real predictions
from the image and a 503 when no artifacts are present.

## Project structure

```
.
├── api/                  # FastAPI service and structured logging
├── app/                  # Streamlit UI
├── models/               # Small artifacts; large weights fetched from the Hub
├── reports/metrics/      # Generated metrics records
├── requirements/         # base, inference-cpu, train, dev
├── src/
│   ├── artifacts.py      # fetches the large weights
│   ├── calibration.py    # temperature scaling
│   ├── config.py         # labels, paths, hyperparameters
│   ├── data.py           # dataset loading
│   ├── embeddings.py     # pretrained GloVe vectors
│   ├── evaluate.py       # evaluation and the results table
│   ├── inference/        # shared predictor and model versioning
│   ├── models/           # Keras architectures
│   ├── training/         # training entry points
│   └── utils/            # preprocessing, metrics, IO
├── scripts/              # publish_weights.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

See [LICENSE](LICENSE).
