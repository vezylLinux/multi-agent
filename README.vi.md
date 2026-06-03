# Hệ thống Lập kế hoạch Du lịch Đa tác nhân

Hệ thống AI lập kế hoạch du lịch cho **Đà Nẵng, Việt Nam** được xây dựng trên kiến trúc đa tác nhân. Người dùng mô tả chuyến đi bằng ngôn ngữ tự nhiên (tiếng Việt hoặc tiếng Anh) và nhận lại lịch trình chi tiết từng ngày kèm khoảng cách thực tế, gợi ý khách sạn và bản đồ tương tác.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Pipeline tác nhân](#pipeline-tác-nhân)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dữ liệu](#dữ-liệu)
- [Cài đặt & Chạy](#cài-đặt--chạy)
- [Tính năng nổi bật](#tính-năng-nổi-bật)
- [API Reference](#api-reference)
- [Sơ đồ](#sơ-đồ)

---

## Tổng quan

Hệ thống nhận đầu vào là một câu mô tả tự do, ví dụ:

> *"Gia đình 2 vợ chồng và 2 bé muốn đi Đà Nẵng 3 ngày 2 đêm. Các bé thích tắm biển, ba mẹ thích hải sản và chùa chiền. Ngân sách tiết kiệm, ở gần biển Mỹ Khê."*

Và trả về:

- **Lịch trình chi tiết từng ngày** (buổi sáng / trưa / chiều / tối)
- **Khoảng cách và thời gian di chuyển thực tế** giữa từng điểm dừng (TrackAsia Directions API)
- **Gợi ý khách sạn phù hợp ngân sách** được chấm điểm theo vị trí + hạng sao + loại hình
- **Bản đồ tương tác** với marker, lộ trình từng ngày và panel thông tin địa điểm
- **Thông tin validation** (khoảng cách leg dài nhất, các vấn đề, đã retry chưa)

---

## Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────┐
│  Frontend  (Next.js :3001)                       │
│  ChatShell · ItineraryFlowPanel · DayMap         │
└────────────────────┬─────────────────────────────┘
                     │  REST / SSE
┌────────────────────▼─────────────────────────────┐
│  Backend  (FastAPI :8000)                        │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  LangGraph Workflow                         │ │
│  │  intake → retrieval → planning →            │ │
│  │  validator → response                       │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  Dịch vụ: Builder · RAG Engine · Formatter      │
│            VRP Optimizer · Scoring · Validator   │
└──────┬───────────────┬──────────────────────────-┘
       │               │
┌──────▼──────┐  ┌─────▼──────────┐
│  SQLite     │  │  Chroma        │
│  travel.db  │  │  2162 vectors  │
│  883 nơi    │  │  (local)       │
└─────────────┘  └────────────────┘
       │
┌──────▼──────────────────────────────┐
│  External APIs                      │
│  OpenRouter (gpt-4o-mini) · OpenAI  │
│  TrackAsia (geocode + routing)      │
│  OpenWeather (tuỳ chọn)             │
└─────────────────────────────────────┘
```

---

## Pipeline tác nhân

### [A1] Intake Agent — Phân tích yêu cầu
Dùng LLM (`temperature=0.0`) để chuyển câu mô tả tự do thành dữ liệu cấu trúc:
- Trích xuất: `destination` (điểm đến), `days` (số ngày), `interests` (sở thích), `budget` (ngân sách), `companion` (thành phần đoàn)
- Suy ra `budget_tier`: `"low"` | `""` | `"high"` để tính điểm khách sạn
- Nếu thiếu thông tin bắt buộc → trả câu hỏi làm rõ thay vì lập kế hoạch

### [A2] Retrieval Agent — Tìm kiếm địa điểm
Xây dựng pool địa điểm ứng viên:
1. Nhúng (embed) query bằng OpenAI `text-embedding-3-small` (dim=1536)
2. Tìm kiếm Chroma vector store bằng cosine similarity + lexical re-ranking (BM25-like)
3. Trả về top-20 địa điểm kèm dữ liệu thời tiết từ TrackAsia / OpenWeather

### [A3] Planning Agent — Lập lịch trình
Xây dựng lịch trình hoàn chỉnh:
1. Tải toàn bộ pool Đà Nẵng từ DB (883 địa điểm, 99.9% có tọa độ)
2. Điền tọa độ còn thiếu cho các địa điểm từ RAG bằng `_enrich_coords()`
3. Chạy **VRP optimization** để phân cụm địa điểm theo địa lý từng ngày
4. Fallback sang **LLM planning** (`temperature=0.3`) nếu VRP không khả dụng
5. Chọn khách sạn bằng **công thức chấm điểm** theo vị trí + hạng sao + loại + ngân sách
6. Gọi TrackAsia Directions API cho từng leg: `distance_km`, `eta_min`, `mode_label`

**Công thức chấm điểm khách sạn:**
```
điểm = 32.0
  + bonus_cùng_quận    (+10 nếu cùng quận với điểm tham quan chính)
  + bonus_sao          (+1.5 mỗi sao)
  + bonus_loại         (resort/villa +2.5 | hostel +1.0)
  + bonus_ngân_sách    (thấp: hostel +5, phạt >2★
                        cao: thương hiệu luxury +6, +4/sao trên 3★)
  - 2.2 × khoảng_cách_trung_bình_km
  - 0.8 × khoảng_cách_lớn_nhất_km
```

### [A4] Validator Agent — Kiểm tra chất lượng
Cổng kiểm tra trước khi trả kết quả cho người dùng:
- **Kiểm tra cấu trúc bằng LLM** (`temperature=0.0`): đủ số ngày, đủ slot sáng/chiều, không quá 5 hoạt động/ngày, không rỗng
- **Kiểm tra khoảng cách**: `too_many_long_legs` (>18 km), `extreme_leg_distance` (>25 km)
- Nếu có lỗi nghiêm trọng → kích hoạt **retry một lần** với `strict_mode=True` và prompt bổ sung ràng buộc
- Lỗi mềm (`too_many_self_service_meals`) được báo cáo nhưng không chặn

### [A5] Response Agent — Tổng hợp & Lưu trữ
Tạo câu trả lời cuối bằng `format_planning_answer()` (formatter.py):
- Ghép: tóm tắt nghiên cứu + lịch trình + khách sạn + thời tiết + mẹo du lịch → văn bản tiếng Việt mạch lạc
- Lưu cuộc trò chuyện, tin nhắn và kế hoạch vào SQLite

---

## Công nghệ sử dụng

| Tầng | Công nghệ |
|---|---|
| Frontend | Next.js 14, TypeScript, CSS |
| Backend | FastAPI, Python 3.11+ |
| Điều phối tác nhân | LangGraph |
| LLM | gpt-4o-mini qua OpenRouter |
| Embedding | text-embedding-3-small (OpenAI) |
| Vector store | Chroma (persistent local) |
| CSDL quan hệ | SQLite |
| Bản đồ & Định tuyến | TrackAsia (geocode + directions + tiles) |
| Thời tiết | OpenWeather API (tuỳ chọn) |
| Tối ưu lộ trình | Custom VRP solver |

---

## Cấu trúc dự án

```
├── app/
│   ├── core/          # Cài đặt, khởi tạo DB, session
│   ├── graph/         # LangGraph nodes, state, edges, LLM calls
│   │   ├── nodes.py   # 5 hàm node tác nhân
│   │   ├── state.py   # TravelGraphState TypedDict (47 trường)
│   │   ├── intake.py  # Prompt trích xuất ý định
│   │   └── llm.py     # generate_answer(), render output
│   ├── itinerary/     # Lập kế hoạch + validation + định dạng
│   │   ├── builder.py      # build_trip_plan_payload(), chấm điểm khách sạn, VRP
│   │   ├── validation.py   # validate_itinerary_plan(), kiểm tra khoảng cách
│   │   ├── formatter.py    # format_planning_answer()
│   │   ├── routing.py      # resolve_location_for_map(), định tuyến leg
│   │   └── vrp.py          # VRP optimizer
│   ├── places/        # RAG engine, scoring, metadata, repository
│   │   ├── vector_rag.py   # retrieve_place_candidates(), re-ranking
│   │   ├── scoring.py      # INTEREST_KEYWORDS (song ngữ VI+EN)
│   │   ├── metadata.py     # enrich_place_record(), infer_intent_tags()
│   │   ├── repository.py   # upsert_places(), list_places()
│   │   └── chroma.py       # Chroma client wrapper
│   └── tools/         # Adapter TrackAsia, Google Places, OpenWeather
├── frontend/
│   ├── components/
│   │   ├── chat-shell.tsx  # UI chính: chat + panel lịch trình
│   │   └── day-map.tsx     # Widget bản đồ TrackAsia
│   ├── services/
│   │   └── api.ts          # REST client (chat, sessions, conversations)
│   └── app/
│       └── globals.css     # Toàn bộ CSS
├── scripts/
│   ├── preprocess.py            # Dữ liệu thô → làm giàu → upsert DB
│   ├── geocode_places.py        # Geocode qua TrackAsia (adaptive rate)
│   ├── ingest_to_chroma.py      # Xây dựng/rebuild Chroma vector index
│   └── test_log.py              # 5-query regression test → logs/
├── data/
│   ├── crawl/processed/         # JSON nguồn (điểm đến, nhà hàng, khách sạn)
│   ├── processed/               # unified_places.json (883 địa điểm Đà Nẵng)
│   ├── chroma/                  # Chroma persistent storage
│   └── travel.db                # SQLite database
├── docs/
│   └── diagrams/                # File PlantUML nguồn
│       ├── system_architecture.puml
│       ├── erd.puml
│       ├── sequence_overview.puml
│       ├── sequence_planning.puml
│       └── sequence_validation.puml
└── logs/                        # Log test hồi quy (test_vi_*.log)
```

---

## Dữ liệu

### Database SQLite (`travel.db`)

| Bảng | Số dòng | Mô tả |
|---|---|---|
| `places` | 883 | Toàn bộ địa điểm Đà Nẵng (99.9% có tọa độ) |
| `place_chunks` | 2162 | Đoạn văn bản để tìm kiếm vector |
| `conversations` | — | Phiên trò chuyện |
| `messages` | — | Tin nhắn user + assistant kèm metadata |
| `plans` | — | Lịch trình đã lưu |
| `principals` | — | Danh tính người dùng (ẩn danh / đã đăng ký) |
| `anonymous_sessions` | — | Phiên cookie-backed |

### Phân loại địa điểm

| Danh mục | Số lượng | Geocoded |
|---|---|---|
| destination (điểm tham quan) | 30 | 100% |
| restaurant (nhà hàng) | 84 | 100% |
| accommodation (khách sạn) | 759 | 99.9% |
| entertainment (giải trí) | 6 | 100% |
| transport (vận chuyển) | 4 | 100% |

### Intent Tags (song ngữ)
`food` · `beach` · `museum` · `heritage` · `spiritual` · `shopping` · `cafe` · `nature` · `nightlife` · `family`

Mỗi tag đều có keyword tiếng Việt (đã loại dấu) và tiếng Anh trong `INTEREST_KEYWORDS` (`scoring.py`).

---

## Cài đặt & Chạy

### Yêu cầu
- Python 3.11+
- Node.js 18+
- API keys: `OPENROUTER_API_KEY`, `TRACKASIA_API_KEY`, `OPENAI_API_KEY` (cho embedding)

### Backend

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Cấu hình biến môi trường
cp .env.example .env
# Chỉnh .env: thêm OPENROUTER_API_KEY, TRACKASIA_API_KEY, v.v.

# 3. (Lần đầu) Tiền xử lý dữ liệu và xây dựng vector index
python scripts/preprocess.py
python scripts/geocode_places.py
python scripts/ingest_to_chroma.py --recreate

# 4. Khởi động backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install

# Cấu hình URL API
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api" > .env.local

npm run dev -- --port 3001
```

### Chạy kiểm tra hồi quy

```bash
python scripts/test_log.py
# Kết quả lưu tại logs/test_vi_<timestamp>.log
```

### Xây dựng lại Chroma sau khi thêm dữ liệu

```bash
python scripts/ingest_to_chroma.py --recreate
```

---

## Tính năng nổi bật

| Tính năng | Chi tiết |
|---|---|
| **Đầu vào song ngữ** | Nhận tiếng Việt (có/không dấu) và tiếng Anh |
| **Chọn khách sạn theo ngân sách** | Thấp → hostel/mini hotel; Cao → resort/thương hiệu luxury |
| **Khoảng cách lộ trình thực tế** | TrackAsia Directions API — khoảng cách + ETA + phương tiện di chuyển |
| **VRP optimization** | Phân cụm địa điểm theo địa lý để giảm thiểu tổng quãng đường |
| **Validation + retry tự động** | Kiểm tra cấu trúc và khoảng cách; tự retry một lần khi thất bại |
| **Bản đồ tương tác** | Marker theo ngày, lộ trình, panel thông tin địa điểm |
| **Output toàn tiếng Việt** | Toàn bộ lịch trình, nhãn và thông báo hệ thống bằng tiếng Việt |
| **Lịch sử hội thoại** | Phiên cookie-backed; lưu cuộc trò chuyện và kế hoạch |
| **Geocoding thích ứng** | 0.15 giây/request với exponential backoff khi gặp rate limit 429 |

---

## API Reference

| Phương thức | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/chat` | Gửi query, nhận ChatResponse đầy đủ |
| `POST` | `/api/chat/stream` | SSE streaming response |
| `POST` | `/api/session/init` | Tạo/tiếp tục phiên (set cookie) |
| `GET` | `/api/session/me` | Thông tin người dùng hiện tại |
| `GET` | `/api/conversations` | Danh sách lịch sử hội thoại |
| `GET` | `/api/conversations/{id}` | Chi tiết cuộc trò chuyện kèm tin nhắn |
| `DELETE` | `/api/conversations/{id}` | Xoá một cuộc trò chuyện |
| `DELETE` | `/api/conversations` | Xoá toàn bộ lịch sử |
| `POST` | `/api/plans/save` | Lưu kế hoạch có cấu trúc |
| `GET` | `/api/plans/{id}` | Lấy kế hoạch đã lưu |
| `POST` | `/api/route/geometry` | Lấy geometry lộ trình cho bản đồ |

### Cấu trúc ChatResponse (các trường chính)

```json
{
  "answer": "KẾ HOẠCH DU LỊCH - ĐÀ NẴNG ...",
  "plan": "LỊCH TRÌNH 3 NGÀY TẠI DA NANG ...",
  "route_plan": [{ "day": 1, "from": "...", "to": "...", "distance_km": 3.5, "eta_min": 8 }],
  "recommended_hotel": { "name": "...", "address": "...", "lat": 16.07, "lon": 108.22 },
  "plan_validation": { "passed": true, "issues": [], "retried": false, "metrics": { "max_leg_km": 9.4 } },
  "collected_info": { "destination": "Da Nang", "days": "3", "interests": "beach, food", "budget": "tiết kiệm" },
  "verified_places": [...],
  "timings": { "intake_ms": 1200, "retrieval_ms": 2100, "planning_ms": 9800, "validator_ms": 3500 }
}
```

---

## Sơ đồ

File PlantUML nguồn trong [`docs/diagrams/`](docs/diagrams/):

| File | Mô tả |
|---|---|
| `system_architecture.puml` | Toàn bộ component + data flow |
| `erd.puml` | Schema database (7 bảng) |
| `sequence_overview.puml` | Luồng tác nhân tổng quan |
| `sequence_planning.puml` | Chi tiết nội bộ planning_node [A3] |
| `sequence_validation.puml` | validator_node [A4] + logic retry |

Render bằng extension [PlantUML cho VS Code](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) (`Alt+D`) hoặc tại [plantuml.com](https://plantuml.com/plantuml/uml/).

---

## Hiệu năng

| Chỉ số | Giá trị |
|---|---|
| Thời gian phản hồi trung bình | ~17 giây |
| Tỉ lệ retry | ~80% các query |
| Độ phủ tọa độ (geocoded) | 99.9% (882 / 883 địa điểm) |
| Số vector Chroma | 2162 chunks |
| Tỉ lệ pass validation (bộ test hồi quy) | 100% (5/5 query) |
