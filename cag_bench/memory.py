
from pathlib import Path
from typing import Dict, Any, List
from .retrieval import tokenize
from .utils import append_jsonl, load_jsonl, extract_terms

class ProjectMemory:
    def __init__(self, path: str | Path, max_chars: int | None = None):
        self.path = Path(path); self.max_chars = max_chars; self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text('', encoding='utf-8')
    def rows(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.path)
    def add(self, task: Dict[str, Any], accepted_summary: str) -> None:
        append_jsonl(self.path, {"task_id":task['id'],"task_title":task['title'],"scope":task.get('scope','project'),"memory_type":task.get('memory_type','decision'),"text":accepted_summary.strip(),"tags":task.get('tags',[]),"promoted_terms":task.get('promote_terms',[])})
    def retrieve(self, task: Dict[str, Any], k: int | None = None) -> List[Dict[str, Any]]:
        rows = self.rows()
        if not rows: return []
        continuity_terms = extract_terms(task.get('continuity_terms', []))
        qtext = ' '.join([task.get('title',''), task.get('prompt',''), ' '.join(task.get('tags',[])), ' '.join(continuity_terms)])
        q = tokenize(qtext); scored = []
        for idx,row in enumerate(rows):
            rtext = ' '.join([row.get('text',''), row.get('scope',''), ' '.join(row.get('tags',[])), ' '.join(row.get('promoted_terms',[]))])
            score = len(q & tokenize(rtext))*3.0 + (idx/max(1,len(rows)))
            if row.get('scope') == 'global': score += 0.5
            scored.append((score,row))
        scored.sort(key=lambda x:x[0], reverse=True)
        chosen=[]; total=0
        limit = len(scored) if k is None else k
        for score,row in scored[:limit]:
            n=len(row.get('text',''))
            if self.max_chars is not None and total+n > self.max_chars and chosen: continue
            chosen.append(row); total += n
        return chosen
