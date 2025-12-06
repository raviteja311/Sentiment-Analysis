# Sentiment-Analysis

A comprehensive sentiment analysis application supporting multiple deep learning models including Logistic Regression, LSTM, GRU, and BERT.

## Features

- **Multiple Models**: Choose between LogisticRegression, LSTM, GRU, and BERT for sentiment classification
- **Pre-trained Models**: All models come pre-trained and ready to use
- **Easy-to-use Interface**: Streamlit-based web UI for text sentiment analysis
- **Fast Inference**: Optimized for quick predictions

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/raviteja311/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. **Create a virtual environment (optional but recommended)**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sentiment-analysis python=3.10
conda activate sentiment-analysis
```

3. **Install dependencies**:
```bash
pip install -r requirements/inference.txt
```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Usage

1. Open the Streamlit app in your browser
2. Select a model from the sidebar (LogisticRegression, LSTM, GRU, or BERT)
3. Enter text in the text area
4. Click "Predict" to analyze sentiment
5. View the sentiment classification and confidence scores

## Supported Models

| Model | Type | Framework | File |
|-------|------|-----------|------|
| LogisticRegression | Shallow | scikit-learn | `models/lr/pipeline.joblib` |
| LSTM | Deep Learning | TensorFlow/Keras | `models/lstm/model_final.keras` |
| GRU | Deep Learning | TensorFlow/Keras | `models/gru/model_final.keras` |
| BERT | Transformer | Hugging Face | `models/bert/` |

## Project Structure

```
Sentiment-Analysis/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── models/
│   ├── bert/                      # Pre-trained BERT model
│   ├── gru/                       # Pre-trained GRU model
│   ├── lstm/                      # Pre-trained LSTM model
│   └── lr/                        # Pre-trained Logistic Regression model
├── src/
│   ├── models/                    # Model wrappers
│   ├── training/                  # Training scripts
│   └── utils/                     # Utility functions
└── requirements/
    ├── inference.txt              # Production dependencies
    └── train.txt                  # Training dependencies
```

## Dependencies

Main dependencies for inference:
- **transformers**: Hugging Face transformer models (for BERT)
- **torch**: PyTorch framework
- **tensorflow**: TensorFlow/Keras for LSTM and GRU
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web UI framework
- **joblib**: Model serialization

## Notes

- All models are pre-trained and ready for inference
- No training data is required to run the application
- The app preprocesses input text automatically
- Sentiment classes: Negative, Neutral, Positive

## License

See LICENSE file for details.

