
import math, re
from dataclasses import dataclass
from pathlib import Path
from typing import List
from .utils import normalize

@dataclass
class SourceChunk:
    source_id: str
    title: str
    text: str

def tokenize(text: str) -> set[str]:
    words = re.findall(r'[a-zA-Z0-9_]+', normalize(text))
    stop = {'the','a','an','and','or','of','to','for','in','on','with','as','is','are','be','this','that','it','by','from','into','use','using','should','must','not','when','then','than'}
    return {w for w in words if len(w) > 2 and w not in stop}

class KeywordRetriever:
    def __init__(self, chunks: List[SourceChunk]):
        self.chunks = chunks
        self.chunk_tokens = [tokenize(c.title + ' ' + c.text) for c in chunks]
    @classmethod
    def from_sources_dir(cls, source_dir: str | Path):
        chunks = []
        for path in sorted(Path(source_dir).glob('*.md')):
            text = path.read_text(encoding='utf-8')
            title = text.splitlines()[0].replace('#','').strip() if text.splitlines() else path.stem
            chunks.append(SourceChunk(path.stem, title, text))
        return cls(chunks)
    def retrieve(self, query: str, k: int = 3) -> List[SourceChunk]:
        q = tokenize(query)
        scored = []
        for c,toks in zip(self.chunks,self.chunk_tokens):
            score = (len(q & toks) / math.sqrt(max(1,len(toks)))) if q else 0.0
            scored.append((score,c))
        scored.sort(key=lambda x:x[0], reverse=True)
        return [c for s,c in scored[:k] if s > 0] or [c for _,c in scored[:k]]
