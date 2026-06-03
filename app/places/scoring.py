from __future__ import annotations

from typing import Any

from app.places.metadata import fold_text, extract_trip_days

# Keyword groups (folded) aligned with intake interests.
INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": (
        "am thuc", "an uong", "dac san", "nha hang", "hai san", "mon an", "quan an",
        "food", "eat", "eating", "cuisine", "seafood", "meal", "dining", "restaurant",
    ),
    "beach": (
        "bien", "tam bien", "bo bien", "bai bien",
        "beach", "sea", "swim", "swimming", "coastal", "shore",
    ),
    "museum": (
        "bao tang", "trien lam", "trung bay",
        "museum", "exhibit", "exhibition", "gallery",
    ),
    "heritage": (
        "di tich", "lich su", "van hoa", "co kinh", "thanh dia",
        "heritage", "historic", "history", "culture", "cultural", "ancient", "old town",
    ),
    "spiritual": (
        "tam linh", "chua", "den", "linh ung", "nha tho",
        "spiritual", "temple", "pagoda", "shrine", "religious", "monastery", "worship",
    ),
    "shopping": (
        "mua sam", "cho dem", "cho",
        "shopping", "shop", "market", "mall", "souvenir", "buy",
    ),
    "cafe": (
        "ca phe", "song ao",
        "cafe", "coffee", "chill", "tea", "relax",
    ),
    "nature": (
        "thien nhien", "nui", "rung", "suoi", "thac", "dao", "leo nui", "doi",
        "nature", "hiking", "outdoor", "forest", "mountain", "waterfall", "island", "eco",
    ),
    "nightlife": (
        "di dem", "quan bar",
        "bar", "pub", "nightlife", "night", "club", "party", "drinks",
    ),
    "family": (
        "gia dinh", "tre em",
        "family", "kid", "kids", "children", "child", "family-friendly",
    ),
}


def _place_blob(place: dict[str, Any]) -> str:
    parts = [
        str(place.get("name") or ""),
        str(place.get("category") or ""),
        str(place.get("description") or ""),
        str(place.get("list_snippet") or ""),
        str(place.get("address") or ""),
    ]
    return fold_text(" ".join(parts))


def _active_interest_groups(query_fold: str) -> list[str]:
    return [tag for tag, kws in INTEREST_KEYWORDS.items() if any(k in query_fold for k in kws)]


def _intent_match_ratio(place: dict[str, Any], place_blob: str, query_fold: str) -> float:
    groups = _active_interest_groups(query_fold)
    cat = str(place.get("category") or "").lower()
    place_tags = {
        str(tag).strip().lower()
        for tag in (place.get("intent_tags") or [])
        if str(tag).strip()
    }
    if not groups:
        toks = [t for t in query_fold.split() if len(t) >= 4]
        if not toks:
            return 0.55
        hits = sum(1 for t in toks if t in place_blob)
        return min(1.0, 0.35 + 0.65 * (hits / max(len(toks), 1)))

    hits = 0.0
    for tag in groups:
        kws = INTEREST_KEYWORDS[tag]
        if tag in place_tags or any(kw in place_blob for kw in kws):
            hits += 1.0
        elif tag == "food" and "restaurant" in cat:
            hits += 1.0
        elif tag == "beach" and "beach" in cat:
            hits += 1.0
        elif tag in {"museum", "heritage"} and "museum" in cat:
            hits += 0.8
    return min(1.0, hits / max(len(groups), 1))


def add_fit_scores(places: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Attach customer_fit_score 0-100 plus intent and retrieval components."""
    if not places:
        return []
    qf = fold_text(query)
    days = extract_trip_days(query) or 1
    n = len(places)
    out: list[dict[str, Any]] = []
    for i, p in enumerate(places):
        rel = p.get("retrieval_relevance")
        if rel is None:
            rel = max(0.28, 1.0 - (i / max(n, 1)) * 0.55)
        rel = max(0.0, min(1.0, float(rel)))
        blob = _place_blob(p)
        intent = _intent_match_ratio(p, blob, qf)
        if str(p.get("category") or "").lower() == "accommodation" and days >= 2:
            intent = max(intent, 0.65)
        raw = 100.0 * (0.42 * rel + 0.58 * intent)
        scored = {
            **p,
            "intent_match_ratio": round(intent * 100, 1),
            "retrieval_relevance_pct": round(rel * 100, 1),
            "customer_fit_score": round(raw, 1),
        }
        out.append(scored)
    return out


def local_catalog_sufficient(ranked: list[dict[str, Any]], top_k: int, min_fit: float) -> bool:
    """True if at least top_k items and each of the top_k meets min_fit."""
    if len(ranked) < top_k:
        return False
    head = ranked[:top_k]
    return all(float(p.get("customer_fit_score") or 0) >= min_fit for p in head)
