from typing import Iterable, List
import re
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")

def basic_clean(text: str) -> str:
    """
    Light text normalization:
    - Lowercase
    - Remove URLs
    - Remove HTML
    - Strip non-alphanumeric (keep spaces)
    - Squash extra spaces
    """
    if text is None:
        return ""
    t = text.lower()
    t = URL_PATTERN.sub(" ", t)
    t = HTML_PATTERN.sub(" ", t)
    t = NON_ALNUM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def batch_clean(texts: Iterable[str]) -> List[str]:
    return [basic_clean(t) for t in texts]