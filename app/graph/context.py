from __future__ import annotations

from typing import Iterable, List


def build_context_from_places(places: Iterable[dict]) -> str:
    chunks: List[str] = []
    for p in places:
        name = p.get("name") or ""
        category = p.get("category") or ""
        address = p.get("address") or ""
        description = p.get("description") or ""
        city = p.get("city") or ""
        rag_snippets = [
            str(snippet).strip()
            for snippet in (p.get("rag_snippets") or [])
            if str(snippet).strip()
        ][:2]

        lines = [
            f"Name: {name}",
            f"Category: {category}",
            f"City: {city}",
            f"Address: {address}",
            f"Description: {description}",
        ]
        if rag_snippets:
            lines.append("RAG snippets:")
            lines.extend([f"- {snippet}" for snippet in rag_snippets])
        score = p.get("customer_fit_score")
        if score is not None:
            lines.append(f"customer_fit_score (0-100): {score}")
            ir = p.get("intent_match_ratio")
            rr = p.get("retrieval_relevance_pct")
            if ir is not None and rr is not None:
                lines.append(f"  intent_match %: {ir}, retrieval_relevance %: {rr}")
        tier = p.get("retrieval_tier")
        if tier:
            lines.append(f"retrieval_tier: {tier}")
        lines.append("---")
        chunks.append("\n".join(lines))
    return "\n".join(chunks).strip()
