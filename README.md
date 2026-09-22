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
| Twitter-RoBERTa | validation | 2000 | 0.7925 | 0.7815 | 0.7363 | 0.7638 | 0.8443 |
| Twitter-RoBERTa | test | 12284 | 0.7077 | 0.7084 | 0.7335 | 0.6920 | 0.6997 |
| Bi-GRU | validation | 2000 | 0.6495 | 0.6326 | 0.5493 | 0.6592 | 0.6891 |
| Bi-GRU | test | 12284 | 0.6175 | 0.5952 | 0.5980 | 0.6573 | 0.5303 |
| Bi-LSTM | validation | 2000 | 0.6520 | 0.6358 | 0.5642 | 0.6688 | 0.6742 |
| Bi-LSTM | test | 12284 | 0.6168 | 0.5921 | 0.5942 | 0.6598 | 0.5223 |
| Logistic Regression | validation | 2000 | 0.6560 | 0.6360 | 0.5364 | 0.6512 | 0.7204 |
| Logistic Regression | test | 12284 | 0.5827 | 0.5795 | 0.6006 | 0.5786 | 0.5592 |

The transformer leads by a clear margin. The two recurrent models are within
half a point of each other - treat them as equivalent - and beat the linear
baseline by roughly 3.5 points of accuracy. See [MODEL_CARD.md](MODEL_CARD.md)
for limitations and intended use.

Each record in `reports/metrics/` stores the hyperparameters, the split sizes and
the class distribution alongside the scores, so any number above can be traced
to what produced it.

## Models

| Key | Model | Framework | Artifact |
|---|---|---|---|
| `lr` | TF-IDF + Logistic Regression | scikit-learn | `models/lr/pipeline.joblib` |
| `lstm` | Bi-LSTM over a learned embedding | Keras 3 | `models/lstm/model_final.keras` |
| `gru` | Bi-GRU over a learned embedding | Keras 3 | `models/gru/model_final.keras` |
| `bert` | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment` | Transformers | `models/bert/` |

The transformer directory is named `bert` for historical reasons; the model is
**RoBERTa** (`model_type: roberta`, byte-pair tokenizer), not BERT.

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
pip install -r requirements/inference-cpu.txt
make fetch-weights                               # 564 MB, see Weights below
```

For a CPU-sized PyTorch install, add
`--extra-index-url https://download.pytorch.org/whl/cpu`.

### Weights

The large weights are **not in git**. They live on the Hugging Face Hub at
[RAVITEJA311/sentiment-analysis-models](https://huggingface.co/RAVITEJA311/sentiment-analysis-models)
and are fetched on demand:

```bash
make fetch-weights                        # all of them, 564 MB
python -m src.artifacts --models lstm     # or just one
```

Point it elsewhere with `SENTIMENT_MODELS_REPO`.

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
| `GET /models` | every model, whether it is available, and why not |
| `POST /predict` | classify one text |
| `POST /predict/batch` | classify up to 256 texts |

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"this is fantastic","model":"bert"}'
```

```json
{
  "model": "bert",
  "label": "positive",
  "confidence": 0.9984,
  "probabilities": {"negative": 0.0005, "neutral": 0.0011, "positive": 0.9984}
}
```

Interactive docs are at `/docs`.

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
docker build --target cpu --build-arg FETCH_MODELS="lstm gru bert" .
```

## Retraining

```bash
pip install -r requirements/train.txt
make train-lr        # ~5 min, CPU
make train-lstm      # ~6 min, CPU
make train-gru       # ~6 min, CPU
make train-bert      # ~2h15m on a GTX 1650 with mixed precision
make evaluate        # scores every available model, regenerates the table above
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
make test         # 131 tests
make lint         # ruff + black
```

Tests skip themselves rather than fail when a model's weights are absent, so the
suite is meaningful on a clone where `make fetch-weights` has not been run. CI runs lint, the test suite on
Python 3.11 and 3.12, and a container smoke test that asserts real predictions
from the image and a 503 when no artifacts are present.

## Project structure

```
.
├── api/                  # FastAPI service
├── app/                  # Streamlit UI
├── models/               # Small artifacts; large weights fetched from the Hub
├── reports/metrics/      # Generated metrics records
├── requirements/         # base, inference-cpu, train, dev
├── src/
│   ├── artifacts.py      # fetches the large weights
│   ├── config.py         # labels, paths, hyperparameters
│   ├── data.py           # dataset loading
│   ├── evaluate.py       # evaluation and the results table
│   ├── inference/        # shared predictor used by API, UI and tests
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
