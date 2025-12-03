from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertWrapper:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.mdl = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.mdl.eval()

    def predict_proba(self, texts, device="cpu", max_length=128):
        enc = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = self.mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
