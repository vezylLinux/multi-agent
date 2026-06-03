"""Run 5 Vietnamese queries and save comprehensive logs with per-leg distances."""
import sys
import io
import json
import httpx
import time
import sqlite3
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

QUERIES = [
    {
        "id": "Q1",
        "label": "Gia đình 3 ngày tiết kiệm",
        "query": (
            "Gia đình tôi gồm 2 vợ chồng và 2 bé (6 và 10 tuổi) muốn đi Đà Nẵng 3 ngày 2 đêm. "
            "Các bé thích tắm biển, vui chơi. Ba mẹ thích thưởng thức hải sản tươi và tham quan chùa chiền. "
            "Ngân sách tiết kiệm, ở gần biển Mỹ Khê."
        ),
    },
    {
        "id": "Q2",
        "label": "Honeymoon cao cấp 5 ngày",
        "query": (
            "Hai vợ chồng mới cưới muốn đi tuần trăng mật tại Đà Nẵng 5 ngày 4 đêm. "
            "Chúng tôi thích ăn nhà hàng sang trọng view biển, tham quan Bà Nà Hills, Ngũ Hành Sơn và chùa chiền. "
            "Ngân sách cao cấp, muốn ở resort 4-5 sao gần biển."
        ),
    },
    {
        "id": "Q3",
        "label": "Solo phượt văn hoá 3 ngày",
        "query": (
            "Tôi đi du lịch một mình 3 ngày 2 đêm tại Đà Nẵng. "
            "Thích khám phá văn hoá địa phương, ăn quán vỉa hè, đạp xe ven biển, "
            "thăm các di tích lịch sử và chùa chiền. "
            "Ngân sách thấp, ưu tiên hostel hoặc khách sạn mini gần trung tâm."
        ),
    },
    {
        "id": "Q4",
        "label": "Nhóm bạn 4 người 4 ngày tầm trung",
        "query": (
            "Nhóm 4 bạn bè du lịch Đà Nẵng 4 ngày 3 đêm. "
            "Chúng tôi thích tắm biển, uống cà phê, ăn đặc sản, khám phá phố đêm và mua sắm. "
            "Ngân sách tầm trung, muốn ở khu vực Sơn Trà hoặc Mỹ Khê gần biển."
        ),
    },
    {
        "id": "Q5",
        "label": "Cán bộ công tác 2 ngày cuối tuần",
        "query": (
            "Tôi là người đi công tác Đà Nẵng, có 2 ngày cuối tuần rảnh rỗi. "
            "Muốn tham quan Bảo tàng Chăm, Thành Điện Hải, ăn hải sản buổi tối. "
            "Thích khách sạn 3 sao, sạch sẽ, gần trung tâm thành phố."
        ),
    },
]


def db_stats():
    conn = sqlite3.connect("data/travel.db")
    cur = conn.cursor()
    def q(sql): cur.execute(sql); return cur.fetchone()[0]
    stats = {
        "total":  q('SELECT COUNT(*) FROM places WHERE city_key="da_nang"'),
        "geo":    q('SELECT COUNT(*) FROM places WHERE city_key="da_nang" AND lat IS NOT NULL AND lat!=0'),
        "dest":   q('SELECT COUNT(*) FROM places WHERE city_key="da_nang" AND category="destination"'),
        "rest":   q('SELECT COUNT(*) FROM places WHERE city_key="da_nang" AND category="restaurant"'),
        "hotel":  q('SELECT COUNT(*) FROM places WHERE city_key="da_nang" AND category="accommodation"'),
        "entert": q('SELECT COUNT(*) FROM places WHERE city_key="da_nang" AND category="entertainment"'),
    }
    conn.close()
    return stats


def run_query(q: dict) -> dict:
    start = time.time()
    resp = httpx.post(
        "http://localhost:8000/api/chat",
        json={"message": q["query"], "top_k": 20, "with_plan": True, "category": None},
        timeout=150,
    )
    elapsed = time.time() - start
    data = resp.json()
    return {"elapsed": elapsed, "data": data}


def build_log(run_ts: str, db: dict, results: list) -> list[str]:
    lines = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    lines += [
        "=" * 80,
        f"TEST RUN (Tiếng Việt có dấu): {run_ts}",
        "=" * 80,
        "",
        "── PROJECT INFO ────────────────────────────────────────────────────────",
        "  Tên dự án    : Multi-Agent Tourism Planning System",
        "  Kiến trúc    : FastAPI + LangGraph (5 agents)",
        "  LLM          : gpt-4o-mini via OpenRouter",
        "  Embedding    : text-embedding-3-small (OpenAI)",
        "  Vector DB    : Chroma (persistent, local)",
        "  Routing      : TrackAsia Directions API",
        "  Geocoding    : TrackAsia Text Search (adaptive rate 0.15s/req)",
        "  Frontend     : Next.js @ http://localhost:3001",
        "  Backend      : FastAPI @ http://localhost:8000",
        "",
        "── AGENT PIPELINE ──────────────────────────────────────────────────────",
        "  intake_node  →  retrieval_node  →  planning_node  →  validator_node  →  response_node",
        "",
        "  intake_node    : LLM extracts destination/days/interests/budget/companion",
        "  retrieval_node : Chroma RAG top-k places + weather",
        "  planning_node  : VRP/LLM builds itinerary + hotel selection (budget-aware)",
        "  validator_node : structural checks + distance checks (too_many_long_legs,",
        "                   extreme_leg_distance, missing_daily_structure, ...)",
        "  response_node  : format final payload",
        "",
        "── HOTEL SCORING FACTORS ────────────────────────────────────────────────",
        "  proximity_to_anchors  : -2.2 × avg_km  -0.8 × max_km",
        "  star_rating_bonus     : +1.5 per star",
        "  type_bonus            : resort/villa +2.5 | hostel/homestay +1.0",
        "  budget_bonus (low)    : budget-type +5 | penalize >2 stars (-3/star above 2)",
        "  budget_bonus (high)   : luxury-brand +6 | +4/star above 3 | -2/star below 3",
        "  dominant_area_bonus   : +10 if hotel in same district as most attractions",
        "",
        "── DATABASE SNAPSHOT ────────────────────────────────────────────────────",
        f"  City             : Đà Nẵng only (Hội An / Quảng Nam removed)",
        f"  Total places     : {db['total']}",
        f"  Geocoded         : {db['geo']} ({db['geo']/db['total']*100:.1f}%)",
        f"  Destinations     : {db['dest']}",
        f"  Restaurants      : {db['rest']}",
        f"  Accommodations   : {db['hotel']}",
        f"  Entertainment    : {db['entert']}",
        "",
        "── VALIDATOR ISSUE TYPES ────────────────────────────────────────────────",
        "  HARD (block + retry) : daily_count_mismatch | missing_daily_structure",
        "                         unrealistic_schedule | too_many_long_legs",
        "                         extreme_leg_distance | empty_place_pool",
        "  SOFT (warn only)     : too_many_self_service_meals",
        "=" * 80,
    ]

    summary_rows = []

    for r in results:
        inp    = r["inp"]
        elapsed= r["elapsed"]
        data   = r["data"]

        pv     = data.get("plan_validation") or {}
        m      = pv.get("metrics", {})
        hotel  = data.get("recommended_hotel") or {}
        col    = data.get("collected_info") or {}
        route  = data.get("route_plan") or []
        plan   = data.get("plan", "") or ""
        timing = data.get("timings") or {}

        has_dist  = sum(1 for l in route if l.get("distance_km") is not None)
        total_km  = sum(l.get("distance_km") or 0 for l in route)
        avg_leg   = total_km / has_dist if has_dist else 0
        max_km    = m.get("max_leg_km", 0)
        long_legs = m.get("long_leg_count_gt_18km", 0)

        summary_rows.append({
            "id": inp["id"], "label": inp["label"],
            "passed": pv.get("passed"), "issues": pv.get("issues", []),
            "retried": pv.get("retried"), "max_km": max_km,
            "legs": f"{has_dist}/{len(route)}", "hotel": hotel.get("name", "N/A"),
            "elapsed": elapsed,
        })

        lines += [
            "",
            "─" * 80,
            f"[{inp['id']}]  {inp['label']}  —  {elapsed:.1f}s",
            "─" * 80,
            "",
            "INPUT:",
            f"  {inp['query']}",
            "",
            "INTAKE EXTRACTED:",
            f"  destination : {col.get('destination','')}",
            f"  days        : {col.get('days','')}",
            f"  interests   : {col.get('interests','')}",
            f"  budget      : {col.get('budget','')}",
            f"  companion   : {col.get('companion','')}",
            "",
            "VALIDATION:",
            f"  passed      : {pv.get('passed')}",
            f"  issues      : {pv.get('issues', [])}",
            f"  retried     : {pv.get('retried')}",
            f"  reason      : {pv.get('reason','')}",
            "",
            "DISTANCE SUMMARY:",
            f"  max_leg_km       : {max_km} km",
            f"  long_legs_gt18km : {long_legs}",
            f"  total_route_km   : {total_km:.1f} km",
            f"  avg_leg_km       : {avg_leg:.1f} km",
            f"  legs_with_dist   : {has_dist}/{len(route)}",
            f"  routing_source   : {m.get('distance_source','')}",
            "",
            "HOTEL:",
            f"  name    : {hotel.get('name','N/A')}",
            f"  address : {hotel.get('address','')}",
            f"  lat     : {hotel.get('lat','?')}",
            f"  lon     : {hotel.get('lon','?')}",
            "",
            "NODE TIMINGS (ms):",
        ]
        for k, v in sorted(timing.items()):
            lines.append(f"  {k:<38} : {v}")

        # ── Per-leg distance table ──────────────────────────────────────────
        lines += [
            "",
            "ROUTE — PER LEG DISTANCES:",
            f"  {'Day':<4} {'Seq':<4} {'Label':<12} {'From':<32} {'→  To':<32} {'km':>6} {'min':>4}  {'Mode':<16} Source",
            f"  {'─'*4} {'─'*4} {'─'*12} {'─'*32} {'─'*32} {'─'*6} {'─'*4}  {'─'*16} {'─'*14}",
        ]
        for leg in route:
            km   = leg.get("distance_km")
            eta  = leg.get("eta_min", "?")
            src  = (leg.get("routing_source") or "?")[:14]
            frm  = leg.get("from", "")[:32]
            to   = leg.get("to", "")[:32]
            day  = str(leg.get("day", "?"))
            seq  = str(leg.get("sequence", "?"))
            lbl  = (leg.get("leg_label") or "")[:12]
            mode = (leg.get("mode_label") or leg.get("recommended_mode") or "")[:16]
            if km is not None:
                lines.append(
                    f"  {day:<4} {seq:<4} {lbl:<12} {frm:<32} {to:<32} {km:>6.2f} {str(eta):>4}  {mode:<16} {src}"
                )
            else:
                lines.append(
                    f"  {day:<4} {seq:<4} {lbl:<12} {frm:<32} {to:<32} {'--':>6} {'--':>4}  {'no coords':<16} --"
                )

        lines += ["", "FULL PLAN:"]
        for pl in plan.split("\n"):
            lines.append("  " + pl)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    lines += [
        "",
        "=" * 80,
        "SUMMARY TABLE",
        "=" * 80,
        f"  {'ID':<4} {'Label':<32} {'Pass':<5} {'Retry':<6} {'MaxKm':>6} {'Legs':<8} {'Time':>6}  Hotel",
        f"  {'─'*4} {'─'*32} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*6}  {'─'*30}",
    ]
    all_passed = True
    total_t = 0.0
    for r in summary_rows:
        p = "✓" if r["passed"] else "✗"
        if not r["passed"]:
            all_passed = False
        total_t += r["elapsed"]
        lines.append(
            f"  {r['id']:<4} {r['label']:<32} {p:<5} {str(r['retried']):<6} "
            f"{r['max_km']:>6.1f} {r['legs']:<8} {r['elapsed']:>5.1f}s  {r['hotel'][:30]}"
        )
    lines += [
        "",
        f"  RESULT    : {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}",
        f"  TOTAL Q   : {len(summary_rows)}",
        f"  AVG TIME  : {total_t/len(summary_rows):.1f}s",
        f"  TOTAL TIME: {total_t:.1f}s",
        "",
        "=" * 80,
        f"END OF LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ]
    return lines


def main():
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Test run: {run_ts} ===")

    db = db_stats()
    print(f"DB: {db['total']} places, {db['geo']} geocoded")

    results = []
    for inp in QUERIES:
        print(f"  [{inp['id']}] {inp['label']}...", end=" ", flush=True)
        r = run_query(inp)
        pv = r["data"].get("plan_validation") or {}
        print(f"passed={pv.get('passed')}  {r['elapsed']:.1f}s")
        results.append({"inp": inp, **r})

    lines = build_log(run_ts, db, results)

    log_path = Path("logs") / f"test_vi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {log_path}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
