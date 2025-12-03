import re
from typing import Optional
import emoji

def preprocess_tweet(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    try:
        text = emoji.demojize(text)
    except Exception:
        pass
    text = re.sub(r"\s+", " ", text).strip()
    return text
