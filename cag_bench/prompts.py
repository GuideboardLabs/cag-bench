
from typing import Dict, Any, List
from .retrieval import SourceChunk
from .utils import coerce_concept_groups

SYSTEM = """You are a careful local coding assistant. Produce concrete implementation guidance. Do not invent architecture that conflicts with established project decisions. Use concise headings and include exact project terms when they matter."""

def format_sources(chunks: List[SourceChunk]) -> str:
    return '\n\n'.join([f"### Source: {c.title}\n{c.text.strip()}" for c in chunks]) if chunks else 'No retrieved sources.'

def format_memory(rows: List[Dict[str, Any]]) -> str:
    if not rows: return 'No prior project memory available.'
    out=[]
    for row in rows:
        tags=', '.join(row.get('tags',[]))
        out.append(f"- [{row.get('scope','project')} | {row.get('memory_type','decision')} | {tags}] {row.get('text','').strip()}")
    return '\n'.join(out)

def rag_prompt(task, sources):
    user=f"""Mode: RAG baseline. You get only the current task and freshly retrieved context. You do not have memory from prior runs.\n\nCurrent project task:\n{task['prompt']}\n\nRetrieved context:\n{format_sources(sources)}\n\nReturn: implementation plan, key files/modules, risks, tests."""
    return [{"role":"system","content":SYSTEM},{"role":"user","content":user}]

def dag_prompt(task, sources):
    user=f"""Mode: DAG baseline. Follow this fixed workflow every time: Plan -> Implement -> Test -> Review -> Approve -> PR summary. You get only the current task and freshly retrieved context. You do not have memory from prior runs.\n\nCurrent project task:\n{task['prompt']}\n\nRetrieved context:\n{format_sources(sources)}\n\nReturn sections: Plan, Implementation, Tests, Review, Approval Gate, PR Summary."""
    return [{"role":"system","content":SYSTEM},{"role":"user","content":user}]

def cag_prompt(task, sources, memory_rows, carry_forward: bool = False):
    carry_forward_block = ""
    if carry_forward and memory_rows:
        carry_forward_block = (
            "Treat the items in project memory as binding prior project decisions. "
            "Before your implementation plan, write a 'Prior decisions carried forward:' bulleted list quoting the memory items you are using. "
            "Do not introduce architecture choices that conflict with prior decisions unless the task explicitly changes them.\n\n"
        )
    user=f"""Mode: CAG. Use current context plus accumulated project memory. The memory reflects accepted decisions from earlier related tasks.\n\nCurrent project task:\n{task['prompt']}\n\nAccumulated project memory:\n{format_memory(memory_rows)}\n\nFreshly retrieved context:\n{format_sources(sources)}\n\nReturn: implementation plan, how this uses prior project decisions, key files/modules, risks, tests."""
    return [{"role":"system","content":SYSTEM + ("\n\n" + carry_forward_block if carry_forward_block else "")},{"role":"user","content":user}]

def _concept_labels(values, limit):
    labels = [g.get("concept", "") for g in coerce_concept_groups(values)]
    labels = [label for label in labels if label]
    return labels[:limit]

def dry_answer(mode: str, task: Dict[str, Any], memory_rows: List[Dict[str, Any]] | None = None) -> str:
    source_terms = task.get('source_evidence_terms', task.get('evidence_terms', []))
    domain_terms = task.get('domain_rule_terms', [])
    evidence_labels = _concept_labels(source_terms, 4) + _concept_labels(domain_terms, 3)
    if not evidence_labels:
        evidence_labels = _concept_labels(task.get('evidence_terms', []), 6)
    lines=[
        f"{mode.upper()} response for {task['title']}.",
        "Checklist: "+', '.join(_concept_labels(task.get('checklist_terms', []), 8)),
        "Evidence: "+', '.join(evidence_labels[:6])
    ]
    if mode == 'cag' and memory_rows:
        mem=[]
        for row in memory_rows: mem.extend(row.get('promoted_terms',[]))
        lines.append('Continuity: '+', '.join(mem[-45:]))
    else:
        lines.append('Continuity: '+', '.join(_concept_labels(task.get('continuity_terms', []), 2)))
    return '\n'.join(lines)
