"""
Comprehensive RAG evaluation for the tourism planning system.

Runs three metric groups in sequence, writes a combined JSON + CSV report
and captures all console output to a log file — everything under one folder.

  1. Retrieval & Grounding  – chunk similarity, local hit rate, intent
                              coverage, whitelist grounding, latency
  2. Itinerary Planning     – validation pass/retry rate, leg distances,
                              issue distribution
  3. Giskard RAGET          – faithfulness, context precision/recall,
                              answer relevancy, answer correctness
                              (per-component scores: Generator, Retriever, KB)

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --output-dir data/test_metrics
    python scripts/eval_rag.py --num-questions 20
    python scripts/eval_rag.py --load-testset data/test_metrics/giskard_testset.json
    python scripts/eval_rag.py --skip-giskard        # no LLM judge needed
    python scripts/eval_rag.py --skip-itinerary

Outputs (all under --output-dir):
    eval_log.txt                  full console transcript
    eval_report.json              complete machine-readable report
    retrieval_per_query.csv       per-query retrieval detail
    itinerary_per_query.csv       per-query itinerary detail
    giskard_per_question.csv      per-question Giskard results (if run)
    giskard_testset.json          generated / loaded testset
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import init_db
from app.core.settings import get_settings
from app.graph.intake import evaluate_intake
from app.itinerary.validation import validate_itinerary_plan
from app.places.metadata import fold_text
from app.places.rag import (
    build_itinerary_artifacts,
    retrieve_trip_artifacts,
)
from app.places.scoring import INTEREST_KEYWORDS, add_fit_scores, local_catalog_sufficient
from app.places.search import MIN_LOCAL_FIT_TO_SKIP_EXTERNAL
from app.places.vector_rag import retrieve_chunk_hits

# ── Test queries ──────────────────────────────────────────────────────────────

_QA_QUERIES = [
    "Có những địa điểm tham quan nổi tiếng nào ở Đà Nẵng?",
    "Nhà hàng hải sản ngon ở Đà Nẵng?",
    "Khách sạn gần biển Mỹ Khê?",
    "Hoạt động giải trí ban đêm ở Đà Nẵng?",
    "Điểm tham quan văn hóa lịch sử tại Đà Nẵng?",
    "Cần làm gì trong 1 ngày ở Đà Nẵng?",
    "Cafe đẹp view biển ở Đà Nẵng?",
    "Địa điểm mua sắm ở Đà Nẵng?",
    "Núi Sơn Trà có gì đặc biệt?",
    "Bảo tàng Chăm ở đâu?",
]

_PLANNING_QUERIES = [
    "lịch trình 3 ngày 2 đêm Đà Nẵng, thích biển và ẩm thực, ngân sách trung bình",
    "lịch trình 2 ngày Đà Nẵng, quan tâm văn hóa lịch sử và mua sắm",
    "lịch trình 4 ngày Đà Nẵng, gia đình có trẻ em, thích thiên nhiên và biển",
    "lịch trình 1 ngày Đà Nẵng, backpacker, thích ẩm thực và cà phê",
    "lịch trình 3 ngày Đà Nẵng, cặp đôi, thích văn hóa và ẩm thực",
]

# ── Log tee: write to stdout + file simultaneously ────────────────────────────

class _Tee:
    def __init__(self, *streams: io.IOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                s.write(data.encode("ascii", errors="replace").decode("ascii"))
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


# ── Display helpers ───────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def _row(label: str, value: Any, width: int = 44) -> None:
    print(f"  {label:<{width}} {value}")


def _pct(lst: list[bool]) -> str:
    if not lst:
        return "N/A"
    return f"{100 * sum(lst) / len(lst):.1f}%"


def _mean_f(lst: list[float], digits: int = 3) -> str:
    return f"{statistics.mean(lst):.{digits}f}" if lst else "N/A"


def _percentile(lst: list[float], p: float) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    return s[min(int(len(s) * p), len(s) - 1)]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── 1. Retrieval & Grounding metrics ─────────────────────────────────────────

def _run_retrieval_metrics(queries: list[str]) -> tuple[dict[str, Any], list[dict]]:
    """
    Returns (summary_dict, per_query_rows).
    per_query_rows is suitable for CSV export.
    """
    chunk_scores: list[float] = []
    chroma_hits: list[bool] = []
    local_sufficient: list[bool] = []
    fit_scores: list[float] = []
    intent_covered: list[bool] = []
    latencies_ms: list[float] = []
    per_query: list[dict] = []

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:60]}")
        t0 = time.perf_counter()

        # Vector chunk retrieval
        chunks = retrieve_chunk_hits(query=query, top_k=6)
        chroma_hit = bool(chunks)
        chroma_hits.append(chroma_hit)
        q_chunk_scores = [float(c.score) for c in chunks]
        chunk_scores.extend(q_chunk_scores)

        # Place retrieval + fit scoring
        result = retrieve_trip_artifacts(query=query, top_k=5)
        scored = add_fit_scores(result.places, query)
        q_fit = [float(p["customer_fit_score"]) for p in scored if p.get("customer_fit_score") is not None]
        fit_scores.extend(q_fit)
        suff = local_catalog_sufficient(scored, top_k=5, min_fit=MIN_LOCAL_FIT_TO_SKIP_EXTERNAL)
        local_sufficient.append(suff)

        # Intent coverage
        qf = fold_text(query)
        active_tags = [tag for tag, kws in INTEREST_KEYWORDS.items() if any(k in qf for k in kws)]
        covered: bool | None = None
        if active_tags:
            place_blobs = [
                fold_text(" ".join([
                    str(p.get("name") or ""),
                    str(p.get("category") or ""),
                    str(p.get("description") or ""),
                    " ".join(str(t) for t in (p.get("intent_tags") or [])),
                ]))
                for p in result.places
            ]
            covered = all(
                any(kw in blob for blob in place_blobs for kw in INTEREST_KEYWORDS[tag])
                for tag in active_tags
            )
            intent_covered.append(covered)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        latencies_ms.append(elapsed_ms)

        per_query.append({
            "query": query,
            "chroma_hit": chroma_hit,
            "local_sufficient": suff,
            "num_chunks": len(chunks),
            "mean_chunk_score": round(statistics.mean(q_chunk_scores), 4) if q_chunk_scores else None,
            "num_places": len(result.places),
            "mean_fit_score": round(statistics.mean(q_fit), 1) if q_fit else None,
            "active_intent_tags": ",".join(active_tags),
            "intent_covered": covered,
            "latency_ms": elapsed_ms,
        })

    summary = {
        "chroma_hit_rate": _pct(chroma_hits),
        "local_catalog_sufficient_rate": _pct(local_sufficient),
        "mean_chunk_similarity_score": _mean_f(chunk_scores, 3),
        "mean_customer_fit_score": _mean_f(fit_scores, 1),
        "intent_coverage_rate": _pct(intent_covered) if intent_covered else "N/A",
        "latency_p50_ms": round(_percentile(latencies_ms, 0.50)),
        "latency_p95_ms": round(_percentile(latencies_ms, 0.95)),
    }
    return summary, per_query


# ── 2. Itinerary planning metrics ─────────────────────────────────────────────

def _run_itinerary_metrics(queries: list[str]) -> tuple[dict[str, Any], list[dict]]:
    pass_flags: list[bool] = []
    retry_flags: list[bool] = []
    issues_dist: dict[str, int] = {}
    max_legs: list[float] = []
    per_query: list[dict] = []

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:60]}")
        t0 = time.perf_counter()

        result = retrieve_trip_artifacts(query=query, top_k=8, with_plan=True)
        artifacts = build_itinerary_artifacts(query=query, places=result.places)
        plan = str(artifacts.get("plan") or "")
        route_plan = artifacts.get("route_plan") or []

        intake = evaluate_intake(query)
        raw_interests = str(intake.collected.get("interests") or "")
        interests = {t.strip().lower() for t in raw_interests.split(",") if t.strip()}

        validation = validate_itinerary_plan(
            query=query,
            plan=plan,
            places=result.places,
            interests=interests,
            route_plan=route_plan,
        )
        passed = bool(validation.get("passed", True))
        pass_flags.append(passed)
        retry_flags.append(not passed)

        q_issues = validation.get("issues") or []
        for issue in q_issues:
            issues_dist[issue] = issues_dist.get(issue, 0) + 1

        metrics = validation.get("metrics") or {}
        max_leg = metrics.get("max_leg_km")
        if isinstance(max_leg, (int, float)) and max_leg > 0:
            max_legs.append(float(max_leg))

        per_query.append({
            "query": query,
            "passed": passed,
            "needs_retry": not passed,
            "issues": "|".join(q_issues),
            "max_leg_km": max_leg,
            "long_leg_count_gt_18km": metrics.get("long_leg_count_gt_18km"),
            "days_expected": metrics.get("days_expected"),
            "reason": validation.get("reason", ""),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    summary = {
        "validation_pass_rate": _pct(pass_flags),
        "retry_trigger_rate": _pct(retry_flags),
        "mean_max_leg_km": round(statistics.mean(max_legs), 1) if max_legs else None,
        "issue_distribution": issues_dist,
    }
    return summary, per_query


# ── 3. Giskard RAGET metrics ──────────────────────────────────────────────────

def _run_giskard_metrics(
    *,
    num_questions: int,
    testset_path: Path,
    settings: Any,
) -> dict[str, Any] | None:
    try:
        import pandas as pd
        import giskard
        from giskard.rag import KnowledgeBase, evaluate, generate_testset
    except ImportError as exc:
        print(f"  [SKIP] giskard not installed ({exc})\n  Install: pip install 'giskard[llm]' pandas")
        return None

    if not (settings.openrouter_api_key or "").strip():
        print("  [SKIP] OPENROUTER_API_KEY not set — Giskard needs an LLM judge.")
        return None

    os.environ.setdefault("OPENAI_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("OPENAI_BASE_URL", settings.openrouter_base_url)
    try:
        giskard.llm.set_llm_model(settings.openrouter_model)
    except Exception:
        pass

    rag_json = ROOT / "data" / "rag" / "rag_documents.json"
    if not rag_json.exists():
        print("  [SKIP] data/rag/rag_documents.json not found — run scripts/build_rag.py first.")
        return None

    raw_docs: list[dict] = json.loads(rag_json.read_text(encoding="utf-8"))
    df = pd.DataFrame([
        {
            "content": d["document_text"],
            "title": d.get("title", ""),
            "category": d.get("category", ""),
        }
        for d in raw_docs
        if str(d.get("document_text") or "").strip()
    ])
    print(f"  Knowledge base: {len(df)} chunks from {rag_json.name}")
    kb = KnowledgeBase(df)

    if testset_path.exists():
        print(f"  Loading testset from {testset_path}")
        from giskard.rag import QATestset
        testset = QATestset.load(str(testset_path))
    else:
        print(f"  Generating {num_questions} questions in Vietnamese (may take a few minutes)…")
        testset = generate_testset(
            kb,
            num_questions=num_questions,
            language="vi",
            agent_description=(
                "Trợ lý du lịch tư vấn địa điểm và lịch trình tại Đà Nẵng, Việt Nam. "
                "Chỉ gợi ý các địa điểm có trong cơ sở dữ liệu. Trả lời bằng tiếng Việt."
            ),
        )
        testset_path.parent.mkdir(parents=True, exist_ok=True)
        testset.save(str(testset_path))
        print(f"  Testset saved -> {testset_path}")

    # Map document_text → Giskard KB index (Giskard uses sequential ints, Chroma uses hashes).
    _doc_text_to_kb_idx: dict[str, int] = {}
    _kb_counter = 0
    for d in raw_docs:
        text = str(d.get("document_text") or "").strip()
        if text:
            if text not in _doc_text_to_kb_idx:
                _doc_text_to_kb_idx[text] = _kb_counter
            _kb_counter += 1

    # place_id → all sibling chunks from raw_docs (for expanding retrieval coverage).
    _place_chunks: dict[str, list[dict]] = {}
    for d in raw_docs:
        pid = str(d.get("place_id") or d.get("doc_id", "").rsplit("_", 1)[0])
        if pid:
            _place_chunks.setdefault(pid, []).append(d)

    def _answer_fn(question: str, history=None):
        # For conversational questions use the last USER turn (has entity name) for retrieval.
        retrieval_query = question
        if history:
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    prev_user = str(turn.get("content") or "").strip()
                    if prev_user:
                        retrieval_query = f"{prev_user} {question}"
                    break

        chunks = retrieve_chunk_hits(query=retrieval_query, top_k=16)

        # For long queries, also retrieve with the first clause (≤80 chars) so that
        # a distracting entity mentioned later doesn't crowd out the main subject.
        if len(retrieval_query) > 80:
            short_query = retrieval_query[:80].rsplit(" ", 1)[0]
            extra_chunks = retrieve_chunk_hits(query=short_query, top_k=8)
            existing_doc_ids = {c.doc_id for c in chunks}
            chunks = chunks + [c for c in extra_chunks if c.doc_id not in existing_doc_ids]

        # Expand: for every matched place, include ALL its sibling chunks so the
        # specific reference chunk (which may rank lower) is still in the returned set.
        seen_texts: set[str] = {c.document_text for c in chunks}
        from app.places.vector_rag import ChunkHit
        extra: list[ChunkHit] = []
        for c in chunks:
            pid = str(c.place_id or c.doc_id.rsplit("_", 1)[0])
            for sibling in _place_chunks.get(pid, []):
                stext = str(sibling.get("document_text") or "").strip()
                if stext and stext not in seen_texts:
                    seen_texts.add(stext)
                    extra.append(ChunkHit(
                        doc_id=str(sibling.get("doc_id") or ""),
                        place_id=str(sibling.get("place_id") or ""),
                        title=str(sibling.get("title") or ""),
                        category=str(sibling.get("category") or ""),
                        city=str(sibling.get("city") or ""),
                        document_text=stext,
                        metadata=dict(sibling.get("metadata") or {}),
                        score=c.score * 0.9,
                    ))
        chunks = chunks + extra
        context_text = "\n".join(
            f"- {c.title}: {c.document_text}" for c in chunks
        )
        answer = "Không có thông tin."
        if (settings.openrouter_api_key or "").strip():
            try:
                from openai import OpenAI as _OAI
                _client = _OAI(
                    base_url=settings.openrouter_base_url,
                    api_key=settings.openrouter_api_key,
                    timeout=25,
                )
                messages: list[dict] = [
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý du lịch. "
                            "Trả lời trực tiếp bằng tiếng Việt, "
                            "chỉ dựa trên thông tin được cung cấp. "
                            "Giữ nguyên các chi tiết cụ thể và cụm từ đặc trưng từ tài liệu, "
                            "không rút gọn, không diễn giải lại. "
                            "Không suy diễn hoặc thêm chi tiết ngoài ngữ cảnh. "
                            "Nếu không có thông tin, trả lời 'Không có thông tin'."
                        ),
                    }
                ]
                if history:
                    for turn in history:
                        role = str((turn or {}).get("role") or "")
                        content = str((turn or {}).get("content") or "").strip()
                        if role in ("user", "assistant") and content:
                            messages.append({"role": role, "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Thông tin:\n{context_text}\n\nCâu hỏi: {question}",
                })
                resp = _client.chat.completions.create(
                    model=settings.openrouter_model,
                    messages=messages,
                    temperature=0.0,
                )
                answer = (resp.choices[0].message.content or "").strip() or answer
            except Exception:
                pass
        try:
            from giskard.rag import AgentAnswer
            from giskard.rag.knowledge_base import Document
            docs = []
            for c in chunks:
                kb_idx = _doc_text_to_kb_idx.get(c.document_text.strip())
                doc_id = kb_idx if kb_idx is not None else c.doc_id
                docs.append(Document(document={"content": c.document_text}, doc_id=doc_id))
            return AgentAnswer(message=answer, documents=docs)
        except Exception:
            return answer

    print("  Running RAGET evaluation…")
    report = evaluate(_answer_fn, testset=testset, knowledge_base=kb)

    correctness: float | None = None
    try:
        correctness = round(float(report.correctness), 1)
    except Exception:
        pass

    component_scores: dict[str, float] = {}
    try:
        # component_scores() returns a DataFrame in Giskard 2.19.x with column "score"
        cs_df = report.component_scores()
        component_scores = {str(k): round(float(v), 3) for k, v in cs_df["score"].to_dict().items()}
    except Exception:
        pass

    by_question_type: dict[str, Any] = {}
    try:
        # correctness_by_question_type() is a method returning DataFrame with column "correctness"
        bq_df = report.correctness_by_question_type()
        by_question_type = {str(k): round(float(v), 3) for k, v in bq_df["correctness"].to_dict().items()}
    except Exception:
        pass

    kb_score: float | None = None
    try:
        kb_score = round(float(report.knowledge_base_score), 1)
    except Exception:
        pass

    per_question: list[dict] = []
    try:
        per_question = report.to_pandas().reset_index().to_dict(orient="records")
    except Exception:
        pass

    return {
        "overall_correctness": correctness,
        "knowledge_base_score": kb_score,
        "component_scores": component_scores,
        "correctness_by_question_type": by_question_type,
        "num_questions_evaluated": len(per_question) or num_questions,
        "per_question": per_question,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full RAG evaluation: retrieval + itinerary + Giskard RAGET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default="data/test_metrics",
        help="Directory for all outputs (default: data/test_metrics)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=20, metavar="N",
        help="Number of Giskard test questions to generate (default: 20)",
    )
    parser.add_argument(
        "--load-testset", metavar="PATH",
        help="Load an existing Giskard testset JSON instead of generating one",
    )
    parser.add_argument(
        "--skip-giskard", action="store_true",
        help="Skip Giskard RAGET (no LLM judge required)",
    )
    parser.add_argument(
        "--skip-itinerary", action="store_true",
        help="Skip itinerary planning section",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "eval_log.txt"
    log_file = log_path.open("w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_file)  # type: ignore[assignment]

    try:
        _run(args, out_dir)
    finally:
        sys.stdout = orig_stdout
        log_file.close()
        print(f"Log saved -> {log_path}")


def _run(args: argparse.Namespace, out_dir: Path) -> None:
    init_db()
    settings = get_settings()

    print(f"Output directory : {out_dir}")
    print(f"Timestamp        : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Embedding model  : {settings.embedding_model}")
    print(f"LLM model        : {settings.openrouter_model}")

    full_report: dict[str, Any] = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.openrouter_model,
    }

    # ── 1. Retrieval & Grounding ──────────────────────────────────────────────
    _section("1. Retrieval & Grounding Metrics")
    print(f"  Queries: {len(_QA_QUERIES)}")
    ret_summary, ret_rows = _run_retrieval_metrics(_QA_QUERIES)

    print()
    for k, v in ret_summary.items():
        _row(k, v)

    full_report["retrieval"] = ret_summary

    csv_ret = out_dir / "retrieval_per_query.csv"
    _write_csv(csv_ret, ret_rows)
    print(f"\n  CSV -> {csv_ret}")

    # ── 2. Itinerary ─────────────────────────────────────────────────────────
    if not args.skip_itinerary:
        _section("2. Itinerary Planning Metrics")
        print(f"  Queries: {len(_PLANNING_QUERIES)}")
        itin_summary, itin_rows = _run_itinerary_metrics(_PLANNING_QUERIES)

        print()
        for k, v in itin_summary.items():
            _row(k, str(v))

        full_report["itinerary"] = itin_summary

        csv_itin = out_dir / "itinerary_per_query.csv"
        _write_csv(csv_itin, itin_rows)
        print(f"\n  CSV -> {csv_itin}")

    # ── 3. Giskard RAGET ─────────────────────────────────────────────────────
    if not args.skip_giskard:
        _section("3. Giskard RAGET Metrics")
        testset_path = (
            Path(args.load_testset) if args.load_testset
            else out_dir / "giskard_testset.json"
        )
        giskard_result = _run_giskard_metrics(
            num_questions=args.num_questions,
            testset_path=testset_path,
            settings=settings,
        )
        if giskard_result:
            print()
            _row("overall_correctness", giskard_result.get("overall_correctness"))
            _row("knowledge_base_score", giskard_result.get("knowledge_base_score"))
            for comp, score in (giskard_result.get("component_scores") or {}).items():
                _row(f"  component [{comp}]", score)
            for qtype, score in (giskard_result.get("correctness_by_question_type") or {}).items():
                _row(f"  by_type [{qtype}]", score)

            per_q = giskard_result.pop("per_question", [])
            full_report["giskard_raget"] = giskard_result

            if per_q:
                csv_gsk = out_dir / "giskard_per_question.csv"
                _write_csv(csv_gsk, per_q)
                print(f"\n  CSV -> {csv_gsk}")
                full_report["giskard_raget"]["per_question"] = per_q

    # ── Save JSON report ──────────────────────────────────────────────────────
    _section("Files Written")
    json_path = out_dir / "eval_report.json"
    json_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _row("eval_report.json", str(json_path))
    for f in sorted(out_dir.iterdir()):
        if f.suffix in {".csv", ".txt", ".json"} and f.name != "eval_report.json":
            _row(f.name, str(f))


if __name__ == "__main__":
    main()
