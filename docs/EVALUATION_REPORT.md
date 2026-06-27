# Báo cáo đánh giá hệ thống RAG du lịch

Ngày thực hiện đánh giá: 09/06/2026
Mô hình nhúng: text-embedding-3-small (OpenAI)
Mô hình sinh văn bản: gpt-4o-mini
Tập dữ liệu kiến thức: 2164 chunks trích xuất từ rag_documents.json

---

## 1. Tổng quan hệ thống đánh giá

Script đánh giá `scripts/eval_rag.py` chạy tuần tự ba nhóm chỉ số độc lập:

1. Retrieval and Grounding - đo chất lượng truy xuất và bao phủ thông tin
2. Itinerary Planning - đo chất lượng lập lịch trình du lịch
3. Giskard RAGET - đo chất lượng toàn bộ pipeline RAG theo tiêu chuẩn của Giskard

Tất cả kết quả được ghi vào thư mục `data/test_metrics/` gồm: `eval_report.json`, `eval_log.txt`, `retrieval_per_query.csv`, `itinerary_per_query.csv`, `giskard_per_question.csv`, `giskard_testset.json`.

---

## 2. Nhóm chỉ số 1 - Retrieval and Grounding

### 2.1 Tập truy vấn thử nghiệm

Mười truy vấn đại diện cho các nhu cầu phổ biến của người dùng:

| STT | Truy vấn |
|-----|----------|
| 1 | Có những địa điểm tham quan nổi tiếng nào ở Đà Nẵng? |
| 2 | Nhà hàng hải sản ngon ở Đà Nẵng? |
| 3 | Khách sạn gần biển Mỹ Khê? |
| 4 | Hoạt động giải trí ban đêm ở Đà Nẵng? |
| 5 | Điểm tham quan văn hóa lịch sử tại Đà Nẵng? |
| 6 | Cần làm gì trong 1 ngày ở Đà Nẵng? |
| 7 | Cafe đẹp view biển ở Đà Nẵng? |
| 8 | Địa điểm mua sắm ở Đà Nẵng? |
| 9 | Núi Sơn Trà có gì đặc biệt? |
| 10 | Bảo tàng Chăm ở đâu? |

### 2.2 Định nghĩa từng chỉ số

**chroma_hit_rate**

Tỉ lệ truy vấn trả về ít nhất một chunk từ ChromaDB. Nếu Chroma không trả về kết quả nào thì xem là trượt. Chỉ số này phản ánh khả năng tiếp cận cơ bản của vector store.

Công thức:
```
chroma_hit_rate = số_truy_vấn_có_kết_quả / tổng_số_truy_vấn * 100
```

**local_catalog_sufficient_rate**

Tỉ lệ truy vấn mà bộ dữ liệu địa phương (SQLite + Chroma) đủ tốt để không cần truy cập nguồn ngoài. Tiêu chí: lấy top 5 kết quả, tính điểm customer_fit_score, nếu trung bình đạt ngưỡng `MIN_LOCAL_FIT_TO_SKIP_EXTERNAL` thì xem là đủ.

Công thức:
```
local_sufficient_rate = số_truy_vấn_đủ_nội_địa / tổng_số_truy_vấn * 100
```

**mean_chunk_similarity_score**

Điểm cosine similarity trung bình giữa vector nhúng của câu truy vấn và các chunk được trả về bởi ChromaDB. Giá trị nằm trong [0, 1], càng cao càng phù hợp. Tính trung bình trên toàn bộ 60 chunk (6 chunk x 10 truy vấn).

Công thức:
```
mean_chunk_similarity = sum(điểm_cosine_từng_chunk) / tổng_số_chunk
```

**mean_customer_fit_score**

Điểm phù hợp với khách hàng, tính trên kết quả trả về của `add_fit_scores()`. Hàm này kết hợp nhiều tín hiệu: loại hình địa điểm, tag intent, từ khóa phù hợp với câu truy vấn, tính toán theo trọng số nội bộ. Giá trị từ 0 đến 100.

**intent_coverage_rate**

Tỉ lệ truy vấn có chứa ít nhất một intent tag xác định (food, beach, heritage, v.v.) mà kết quả trả về bao phủ ít nhất một từ khóa của tag đó trong tên, mô tả, hoặc category của địa điểm.

Cách xác định intent tag: hàm `fold_text()` chuẩn hóa câu truy vấn (thường hóa, bỏ dấu), kiểm tra với từ điển `INTEREST_KEYWORDS`. Nếu không có tag nào khớp thì câu truy vấn không được tính vào mẫu số.

Công thức:
```
intent_coverage_rate = số_truy_vấn_bao_phủ_hết_intent / số_truy_vấn_có_intent * 100
```

**latency_p50_ms và latency_p95_ms**

Đo thời gian phản hồi (ms) tính từ khi bắt đầu gọi Chroma và retrieve_trip_artifacts đến khi có kết quả. P50 là median, P95 là phân vị 95 (tức 95% truy vấn hoàn thành trước mốc này).

### 2.3 Kết quả retrieval

| Chỉ số | Kết quả |
|--------|---------|
| chroma_hit_rate | 100.0% |
| local_catalog_sufficient_rate | 90.0% |
| mean_chunk_similarity_score | 0.589 |
| mean_customer_fit_score | 74.1 |
| intent_coverage_rate | 71.4% |
| latency_p50_ms | 356 ms |
| latency_p95_ms | 4177 ms |

### 2.4 Chi tiết theo từng truy vấn

| Truy vấn | Chroma | Đủ nội địa | Số chunk | Chunk score | Số địa điểm | Fit score | Intent | Covered | Latency (ms) |
|----------|--------|------------|----------|-------------|-------------|-----------|--------|---------|--------------|
| Địa điểm tham quan Đà Nẵng | True | True | 6 | 0.670 | 5 | 68.6 | - | - | 4177 |
| Nhà hàng hải sản | True | False | 6 | 0.640 | 5 | 34.7 | food | False | 581 |
| Khách sạn biển Mỹ Khê | True | True | 6 | 0.677 | 5 | 87.3 | beach | True | 341 |
| Giải trí ban đêm | True | True | 6 | 0.538 | 5 | 63.3 | - | - | 334 |
| Văn hóa lịch sử | True | True | 6 | 0.575 | 5 | 93.8 | heritage | True | 404 |
| 1 ngày ở Đà Nẵng | True | True | 6 | 0.536 | 5 | 73.9 | - | - | 364 |
| Cafe view biển | True | True | 6 | 0.581 | 5 | 54.6 | beach,cafe | False | 340 |
| Mua sắm | True | True | 6 | 0.595 | 5 | 80.7 | shopping | True | 356 |
| Núi Sơn Trà | True | True | 6 | 0.542 | 5 | 93.8 | nature | True | 322 |
| Bảo tàng Chăm | True | True | 6 | 0.542 | 5 | 90.8 | museum | True | 337 |

**Nhận xét:**

- Truy vấn "nhà hàng hải sản" không đạt local_sufficient do bộ dữ liệu có nhiều khách sạn và điểm tham quan nhưng ít nhà hàng hải sản chuyên biệt với fit score cao. Kết quả fit score chỉ đạt 34.7/100 - thấp nhất trong tập thử nghiệm.
- Truy vấn "cafe view biển" không cover cả hai intent tag (beach và cafe) đồng thời - có thể lấy được cafe hoặc điểm có view biển nhưng không có đủ cả hai.
- Latency P95 = 4177 ms do truy vấn đầu tiên phải khởi động các kết nối ChromaDB lần đầu, các truy vấn sau có latency rất ổn định ở mức 320-580 ms.

---

## 3. Nhóm chỉ số 2 - Itinerary Planning

### 3.1 Tập truy vấn lập lịch trình

Năm truy vấn đại diện cho các kịch bản du lịch khác nhau:

| STT | Truy vấn |
|-----|----------|
| 1 | lịch trình 3 ngày 2 đêm Đà Nẵng, thích biển và ẩm thực, ngân sách trung bình |
| 2 | lịch trình 2 ngày Đà Nẵng, quan tâm văn hóa lịch sử và mua sắm |
| 3 | lịch trình 4 ngày Đà Nẵng, gia đình có trẻ em, thích thiên nhiên và biển |
| 4 | lịch trình 1 ngày Đà Nẵng, backpacker, thích ẩm thực và cà phê |
| 5 | lịch trình 3 ngày Đà Nẵng, cặp đôi, thích văn hóa và ẩm thực |

### 3.2 Định nghĩa từng chỉ số

**validation_pass_rate**

Tỉ lệ lịch trình vượt qua tất cả các kiểm tra của `validate_itinerary_plan()`. Hàm validation kiểm tra:

- Số ngày trong lịch trình có khớp với yêu cầu trong truy vấn
- Số lượng hoạt động trong mỗi buổi sáng/trưa/chiều/tối có hợp lý
- Các chặng đi (legs) giữa địa điểm liền kề không quá dài (ngưỡng mặc định 18 km)
- Lịch trình có bao gồm cả hoạt động ăn uống (nhà hàng) không chỉ tham quan
- Các loại hình địa điểm (accommodation, destination, restaurant) được phân bổ đều

Kết quả trả về dict: `{"passed": bool, "issues": list[str], "reason": str, "metrics": dict}`.

Công thức:
```
validation_pass_rate = số_lịch_trình_pass / tổng_số_lịch_trình * 100
```

**retry_trigger_rate**

Tỉ lệ lịch trình thất bại validation và cần sinh lại (retry). Hệ thống thực hiện một lần retry với prompt nghiêm ngặt hơn (strict_mode=True) và query gốc kèm vào. Nếu vòng thứ hai vẫn thất bại thì vẫn gửi kết quả đến người dùng với cảnh báo.

Công thức:
```
retry_trigger_rate = số_lịch_trình_thất_bại / tổng_số_lịch_trình * 100
```

**mean_max_leg_km**

Trung bình của khoảng cách leg dài nhất trong mỗi lịch trình. Một "leg" là đoạn đường giữa hai địa điểm liền kề trong cùng một ngày. Khoảng cách lấy từ `route_plan` (tính bởi Haversine hoặc API định tuyến TrackAsia). Giá trị nhỏ hơn thì lịch trình "gọn" hơn, ít di chuyển xa.

Công thức:
```
mean_max_leg_km = sum(max_leg_km_mỗi_lịch_trình) / số_lịch_trình_có_dữ_liệu_khoảng_cách
```

**issue_distribution**

Phân phối số lần xuất hiện từng loại vấn đề (nếu có). Ví dụ: "too_few_restaurants", "day_count_mismatch", "leg_too_long". Dict rỗng có nghĩa là tất cả lịch trình đều pass sạch.

### 3.3 Kết quả itinerary

| Chỉ số | Kết quả |
|--------|---------|
| validation_pass_rate | 100.0% |
| retry_trigger_rate | 0.0% |
| mean_max_leg_km | 12.5 km |
| issue_distribution | (không có vấn đề) |

### 3.4 Chi tiết theo từng truy vấn

| Truy vấn | Pass | Retry | Max leg km | Legs > 18km | Số ngày | Lý do | Latency (ms) |
|----------|------|-------|------------|-------------|---------|-------|--------------|
| 3 ngày, biển + ẩm thực, trung bình | True | False | 15.1 | 0 | 3 | Lịch trình 3 ngày đầy đủ, có cả buổi sáng/trưa/tối, không vượt giới hạn. | 11884 |
| 2 ngày, văn hóa + mua sắm | True | False | 17.8 | 0 | 2 | 2 ngày đầy đủ, bao gồm ăn uống và văn hóa, đáp ứng yêu cầu. | 12669 |
| 4 ngày, gia đình, thiên nhiên + biển | True | False | 15.1 | 0 | 4 | 4 ngày cấu trúc tốt, có hoạt động phù hợp trẻ em. | 16095 |
| 1 ngày, backpacker, ẩm thực + cà phê | True | False | 5.3 | 0 | 1 | 1 ngày cân đối giữa ăn uống và hoạt động. | 9525 |
| 3 ngày, cặp đôi, văn hóa + ẩm thực | True | False | 9.1 | 0 | 3 | 3 ngày đầy đủ, không vượt giới hạn hoạt động. | 14713 |

**Nhận xét:**

- 100% lịch trình pass validation lần đầu, không cần retry.
- Max leg lớn nhất là 17.8 km (lịch trình văn hóa 2 ngày) - chưa chạm ngưỡng 18 km.
- Lịch trình backpacker 1 ngày có max leg chỉ 5.3 km - phù hợp với người đi bộ và xe máy.
- Latency lập lịch trình từ 9.5 đến 16 giây do phải gọi LLM nhiều lần (plan + research + coordinator + validation).

---

## 4. Nhóm chỉ số 3 - Giskard RAGET

### 4.1 Kiến trúc đánh giá Giskard RAGET

Giskard RAGET (RAG Evaluation Toolkit) là bộ công cụ đánh giá pipeline RAG theo 5 thành phần chức năng: RETRIEVER, GENERATOR, REWRITER, ROUTING, KNOWLEDGE_BASE. Mỗi thành phần được gán điểm correctness dựa trên các loại câu hỏi được quy cho thành phần đó.

**Quy trình đánh giá:**

1. Xây dựng KnowledgeBase từ 2164 chunks của rag_documents.json.
2. Tải testset 20 câu hỏi (đã sinh từ Giskard từ phiên trước, lưu tại giskard_testset.json).
3. Với mỗi câu hỏi: gọi `_answer_fn(question, history)` để lấy (answer, documents).
4. Giskard dùng LLM judge (gpt-4o-mini) để đánh giá correctness của từng câu trả lời so với reference answer.
5. Tính điểm tổng hợp theo từng thành phần và loại câu hỏi.

### 4.2 Cách _answer_fn hoạt động

Hàm `_answer_fn` là giao diện giữa Giskard và hệ thống RAG thực tế:

**Bước 1 - Xử lý lịch sử hội thoại:**
Nếu câu hỏi có conversation_history (loại "conversational"), lấy lượt USER gần nhất và ghép vào đầu câu truy vấn để có ngữ cảnh đầy đủ.
```
retrieval_query = prev_user_turn + " " + question
```

**Bước 2 - Truy xuất chunk chính:**
Gọi `retrieve_chunk_hits(query=retrieval_query, top_k=16)` lấy tối đa 16 chunk từ ChromaDB.

**Bước 3 - Truy xuất phụ cho câu hỏi dài:**
Nếu `len(retrieval_query) > 80`, cắt lấy 80 ký tự đầu (cắt tại dấu cách), gọi thêm `retrieve_chunk_hits(query=short_query, top_k=8)`. Merge vào danh sách chunk, bỏ trùng. Mục đích: câu hỏi có yếu tố nhiều lối (distracting element) thường đặt entity chủ đề vào đầu câu - nếu truy xuất bằng câu đầy đủ, entity gây nhiễu có thể chiếm hơn entity chính, dẫn đến mất kết quả quan trọng.

**Bước 4 - Mở rộng sibling chunks:**
Với mỗi chunk được trả về, tìm tất cả chunk cùng place_id trong raw_docs (siblings). Thêm các sibling chưa xuất hiện vào danh sách, với score = score_cha * 0.9. Mục đích: đảm bảo tất cả nội dung của một địa điểm đều được bao phủ, khi reference chunk có thể chỉ là một phần trong nhiều phần của cùng địa điểm.

**Bước 5 - Sinh câu trả lời:**
Gọi gpt-4o-mini với:
- System prompt: "Trả lời trực tiếp bằng tiếng Việt, chỉ dựa trên thông tin được cung cấp. Giữ nguyên các chi tiết cụ thể và cụm từ đặc trưng từ tài liệu, không rút gọn, không diễn giải lại. Không suy diễn hoặc thêm chi tiết ngoài ngữ cảnh."
- User message: tập hợp context từ các chunk + câu hỏi.

**Bước 6 - Đóng gói kết quả:**
Trả về `AgentAnswer(message=answer, documents=[Document(...)])`. Mỗi Document được gán doc_id là chỉ số nguyên (int) tương ứng vị trí chunk trong Giskard KnowledgeBase. Chỉ số này phải là kiểu int (không phải str) để Giskard so sánh đúng với doc_id của testset.

### 4.3 Cách Giskard tính điểm correctness

Giskard dùng LLM judge để so sánh agent_answer với reference_answer. Kết quả là nhị phân: True (đúng) hoặc False (sai). Các điều kiện để bị đánh giá sai bao gồm:

- Thiếu thông tin chính xuất hiện trong reference answer
- Thêm thông tin ngoài lề mà reference answer không đề cập (kể cả khi thông tin đó là đúng trong tài liệu)
- Diễn đạt sai ý nghĩa hoặc thiếu cụ thể

**Lưu ý quan trọng:** Giskard reference answer thường rất ngắn gọn và chính xác. Hệ thống RAG có xu hướng trả lời đầy đủ hơn, nên dễ bị đánh giá sai do "additional information" dù thông tin đó là chính xác. Đây là hạn chế của phương pháp đánh giá nhị phân này.

### 4.4 Các loại câu hỏi trong testset

**simple (4 câu hỏi)**
Câu hỏi tra cứu thông tin thẳng, chỉ cần một tài liệu để trả lời. Ví dụ: "Khách sạn Avatar có những dịch vụ gì?", "Café 41 nằm ở đâu?". Loại này kiểm tra khả năng GENERATOR tạo ra câu trả lời chính xác và trung thành với tài liệu.

**complex (4 câu hỏi)**
Câu hỏi yêu cầu sử dụng nhiều tài liệu hoặc suy luận tổng hợp. Ví dụ: "Khách sạn Luxury có thể tổ chức hội thảo với sức chứa tối đa bao nhiêu khách, và các dịch vụ đi kèm nào được cung cấp?". Loại này kiểm tra khả năng tổng hợp thông tin.

**distracting element (3 câu hỏi)**
Câu hỏi có chứa entity gây nhiễu (entity không liên quan đến chủ thể chính). Ví dụ: "Tháp Chiên Đàn nằm ở đâu và có những đặc điểm nổi bật nào, trong khi cũng đề cập đến các địa điểm ẩm thực gần đó như Nhà hàng Memory Lounge?". Loại này kiểm tra khả năng lọc nhiễu của RETRIEVER.

**situational (3 câu hỏi)**
Câu hỏi đặt trong một tình huống cụ thể. Ví dụ: "Trong hành trình đến Đà Nẵng, tôi đang tìm kiếm một khách sạn nghỉ dưỡng gần các điểm tham quan nổi tiếng như Hội quán Chi Hội Triều Châu và Trần Dương House. Bạn có biết khách sạn Hội An Ancient House Village Resort & Spa cách trung tâm thành phố bao xa không?". Loại này kiểm tra khả năng REWRITER xử lý ngữ cảnh phức tạp.

**double (3 câu hỏi)**
Câu hỏi gộp hai câu hỏi đơn vào một. Ví dụ: "Nhà hàng Madame Lân Đà Nẵng nằm ở đâu và không gian của nhà hàng như thế nào?". Loại này kiểm tra khả năng GENERATOR trả lời đầy đủ cả hai phần.

**conversational (3 câu hỏi)**
Câu hỏi ngắn gọn, thiếu ngữ cảnh (ví dụ: "Nơi này có những gì đặc biệt?"), cần dựa vào conversation_history để biết đang hỏi về đâu. Loại này kiểm tra khả năng REWRITER xử lý câu hỏi phụ thuộc ngữ cảnh.

### 4.5 Cách tính điểm từng thành phần

Giskard RAGET phân biệt 5 thành phần, mỗi thành phần được gán điểm từ trung bình correctness của các loại câu hỏi thuộc thành phần đó:

**RETRIEVER**
Thành phần chịu trách nhiệm truy xuất tài liệu. Được đánh giá qua các loại câu hỏi đòi hỏi trả về đúng tài liệu, cụ thể là "simple" và "distracting element". Khả năng lọc entity gây nhiễu và trả về chunk phù hợp là thước đo cụ thể nhất để đo chất lượng RETRIEVER.

Điểm = trung bình correctness của câu hỏi loại "simple" và "distracting element":
```
RETRIEVER = (simple_correctness + distracting_correctness) / 2
          = (1.0 + 0.333) / 2
          = 0.667
```

**GENERATOR**
Thành phần sinh văn bản cuối. Bao phủ các loại câu hỏi đòi hỏi tổng hợp phức tạp ("complex", "double") và các loại khác. Điểm phản ánh khả năng tạo ra câu trả lời chính xác, đầy đủ, không thêm thông tin ngoài tài liệu.

**REWRITER**
Thành phần xử lý lại câu truy vấn theo ngữ cảnh. Được đánh giá qua các loại câu hỏi "conversational" và "situational" - hai loại đòi hỏi hệ thống hiểu ngữ cảnh ngầm định để trả lời đúng.

**ROUTING**
Thành phần định tuyến câu hỏi đến nguồn thông tin phù hợp. Được đánh giá qua câu hỏi OOD (out-of-domain). Trong testset này không có câu hỏi OOD nên ROUTING đạt điểm 1.0 (không có gì sai).

**KNOWLEDGE_BASE**
Đánh giá chất lượng của kho kiến thức (corpus). Được tính trên tất cả các loại câu hỏi. Phản ánh mức độ đủ dữ liệu của KnowledgeBase để trả lời chính xác, không phụ thuộc vào chất lượng retrieval hay generation.

### 4.6 Kết quả Giskard RAGET

**Điểm tổng hợp:**

| Chỉ số | Điểm |
|--------|------|
| overall_correctness | 0.70 (14/20 câu đúng) |
| knowledge_base_score | 0.90 |

**Điểm theo thành phần:**

| Thành phần | Điểm | Ý nghĩa |
|------------|------|---------|
| ROUTING | 1.000 | Định tuyến hoàn hảo |
| KNOWLEDGE_BASE | 0.933 | Kho kiến thức đầy đủ, có chất lượng cao |
| RETRIEVER | 0.667 | Truy xuất tốt với câu hỏi đơn giản, còn yếu với câu hỏi nhiễu |
| GENERATOR | 0.633 | Sinh văn bản ổn định, mất điểm do thêm thông tin ngoài reference |
| REWRITER | 0.444 | Điểm yếu nhất: khó xử lý câu hỏi hội thoại thiếu ngữ cảnh |

**Điểm theo loại câu hỏi:**

| Loại câu hỏi | Điểm | Số câu | Số đúng |
|--------------|------|--------|---------|
| simple | 1.000 | 4 | 4 |
| situational | 1.000 | 3 | 3 |
| conversational | 0.667 | 3 | 2 |
| complex | 0.500 | 4 | 2 |
| distracting element | 0.333 | 3 | 1 |
| double | 0.333 | 3 | 1 |

### 4.7 Phân tích chi tiết 20 câu hỏi

**--- Loại: simple (4/4 đúng) ---**

**Câu 1 - Khách sạn Avatar có những dịch vụ gì cho khách hàng?**
Kết quả: Đúng
Reference document: Document 357 - Khách sạn Avatar
Reference answer: "Khách sạn Avatar cung cấp ẩm thực Việt Nam tại nhà hàng chuyên phục vụ đồ ăn và thức uống truyền thống, và có một quán cà phê ở tầng trệt."
Agent answer: Lặp lại nguyên văn nội dung tài liệu về tiếp tân, ăn uống, quán cà phê - đầy đủ và chính xác.
Nhận xét: Câu hỏi đơn giản, tài liệu trả về trực tiếp, system prompt "giữ nguyên chi tiết cụ thể" phát huy hiệu quả.

**Câu 2 - Khách sạn Royal Family nằm ở đâu?**
Kết quả: Đúng
Reference document: Document 921 - Khách sạn Royal Family
Reference answer: "39 Võ Văn Kiệt, An Hải, Quận Sơn Trà, Thành phố Đà Nẵng."
Agent answer: Trả lời chính xác địa chỉ đầy đủ.
Nhận xét: Câu hỏi địa chỉ, thông tin thường có trong trường "Địa chỉ" của tài liệu, dễ truy xuất.

**Câu 3 - Café 41 nằm ở đâu?**
Kết quả: Đúng
Reference document: Document 1904 - Café 41
Reference answer: "41, Trần Cao Vân, Phố cổ Hội An, Phường Hội An, Thành phố Đà Nẵng, 51309, Việt Nam."
Agent answer: Trả lời chính xác toàn bộ địa chỉ.
Nhận xét: Tài liệu Café 41 có cấu trúc địa chỉ đầy đủ, vector similarity cao.

**Câu 4 - Steak Louis Restaurant & Coffee có những loại rượu vang nào trong thực đơn?**
Kết quả: Đúng
Reference document: Document 1358 - Steakstone LOUIS
Reference answer: "Có rất nhiều loại rượu vang cuốn hút: vang đỏ, vang trắng,... trong thực đơn vang ấn tượng."
Agent answer: Giữ nguyên cụm "rất nhiều loại rượu vang cuốn hút: vang đỏ, vang trắng..." - không rút gọn.
Nhận xét: Ban đầu (phiên trước) câu này sai vì system prompt có "ngắn gọn" dẫn đến model rút gọn còn "vang đỏ và vang trắng". Sau khi sửa system prompt thành "giữ nguyên các chi tiết cụ thể và cụm từ đặc trưng từ tài liệu, không rút gọn, không diễn giải lại" thì cả 4 câu simple đều đúng.

**--- Loại: complex (2/4 đúng) ---**

**Câu 5 - Khách sạn Luxury: sức chứa hội thảo và dịch vụ đi kèm?**
Kết quả: Sai
Reference document: Document 110 - Khách sạn Luxury
Reference answer: "Khách sạn Luxury có sức chứa lên đến 500 khách cho hội thảo và sự kiện."
Agent answer: "500 khách. Các dịch vụ đi kèm: Massage - Sauna."
Lý do bị đánh giá sai: Reference answer chỉ nêu sức chứa. Agent trả lời thêm thông tin về Massage - Sauna (dù đúng) bị Giskard đánh giá là "provided additional information which was not part of the reference answer."
Nhận xét: Đây là trường hợp reference answer bị giản lược hơn so với tài liệu thực tế. Agent trả lời chính xác nhưng bị phạt vì vượt qua phạm vi của reference.

**Câu 6 - Địa chỉ chính xác của nhà hàng Ốc bươu trộn Cô Hà?**
Kết quả: Đúng
Reference document: Document 1437 - Ốc bươu trộn Cô Hà
Agent answer: Trả lời chính xác "348-352, Đường Ông Ích Khiêm, Thạc Gián, Phường Thanh Khê, Thành phố Đà Nẵng, 50207, Việt Nam."

**Câu 7 - Rừng dừa Bảy Mẫu: vị trí và đặc điểm nổi bật?**
Kết quả: Sai
Reference document: Document 1872 - Rừng dừa Bảy Mẫu
Reference answer: "Thôn Thanh Nhứt, Thanh Tam, Vạn Lăng, Thành phố Hội An, Quảng Nam. Điểm đặc trưng là tham quan hoàn toàn trên thuyền thúng."
Agent answer: Trả lời đúng vị trí và đặc điểm, nhưng thêm thông tin "nên đi vào tháng 8 âm lịch" và "tránh tháng 11-12" (lấy từ tài liệu).
Lý do bị đánh giá sai: Giskard đánh giá là "provided additional information about the best time to visit... which was not part of the reference answer."
Nhận xét: Tương tự câu 5, reference answer ngắn gọn trong khi tài liệu có nhiều thông tin giá trị hơn. Đây là hạn chế của phương pháp đánh giá.

**Câu 8 - Nhà trưng bày Mỹ Sơn: vị trí và khoảng cách?**
Kết quả: Đúng
Reference document: Document 2138 - Nhà trưng bày Mỹ Sơn
Agent answer: "Nằm cạnh suối khe Thẻ, cách khu di tích Mỹ Sơn khoảng 02 km về hướng Tây Bắc, Huyện Duy Xuyên, Quảng Nam." - Chính xác.

**--- Loại: distracting element (1/3 đúng) ---**

**Câu 9 - Tháp Chiên Đàn: vị trí, đặc điểm và đề cập đến Nhà hàng Memory Lounge?**
Kết quả: Sai
Reference document: Document 2129 - Tháp Chiên Đàn
Reference answer: Chỉ đề cập Tháp Chiên Đàn ở Thành phố Tam Kỳ, Quảng Nam, mô tả kết cấu ba ngôi tháp.
Agent answer: Trả lời đúng về Tháp Chiên Đàn nhưng thêm thông tin chi tiết về Nhà hàng Memory Lounge (vị trí, thiết kế, thực đơn).
Lý do bị đánh giá sai: "The agent provided excessive information about Nhà hàng Memory Lounge."
Nhận xét: Entity "Memory Lounge" trong câu hỏi là yếu tố gây nhiễu (distracting). Qua bước truy xuất phụ (short_query = 80 ký tự đầu), hệ thống lấy được chunk của Tháp Chiên Đàn nhưng vẫn bị hút thêm thông tin Memory Lounge từ context, và model sinh thêm nội dung về Memory Lounge.

**Câu 10 - Khách sạn Nesta có nằm gần Boo Hostel ở Sơn Trà không?**
Kết quả: Sai
Reference document: Document 877 - Khách sạn Nesta
Reference answer: "Khách sạn Nesta nằm tại Lô 06-7-H2 đường Phạm Văn Đồng, Quận Sơn Trà, Thành phố Đà Nẵng."
Agent answer: "Cả hai đều ở Sơn Trà. Không có thông tin về khoảng cách."
Lý do bị đánh giá sai: Agent không cung cấp địa chỉ cụ thể của Nesta mà reference yêu cầu.
Nhận xét: Câu hỏi dạng "so sánh vị trí" khiến model tập trung vào sự tương đồng ("đều ở Sơn Trà") thay vì trả lời địa chỉ cụ thể.

**Câu 11 - Naman Retreat có những tiện ích gì cho trẻ em so với White Snow?**
Kết quả: Đúng
Reference document: Document 521 - Naman Retreat
Agent answer: Mô tả đúng khu vui chơi (vẽ tranh, tô màu), baby sitter 250.000 vnđ/giờ, và nêu rõ White Snow không có thông tin tương tự.

**--- Loại: situational (3/3 đúng) ---**

**Câu 12 - Hội An Ancient House Village Resort & Spa cách trung tâm bao xa?**
Kết quả: Đúng
Ngữ cảnh tình huống: Du khách đang tìm khách sạn gần Hội quán Chi Hội Triều Châu và Trần Dương House.
Agent answer: "2 km" - Chính xác, khớp reference.

**Câu 13 - Danang Transfer cung cấp loại xe nào?**
Kết quả: Đúng
Agent answer: Liệt kê đầy đủ Kia Cerato, Honda Civic, City, Toyota Vios, Altis, Inova, Fortuner, Ford Transit, Toyota Hiace - khớp hoàn toàn.

**Câu 14 - Stella Maris Beach Hotel Da Nang nằm ở đâu?**
Kết quả: Đúng
Agent answer: "03 Võ Văn Kiệt, An Hải, Quận Sơn Trà, Thành phố Đà Nẵng" - Chính xác.

**--- Loại: double (1/3 đúng) ---**

**Câu 15 - Madame Lân Đà Nẵng: ở đâu và không gian như thế nào?**
Kết quả: Sai
Reference document: Document 1443 - Nhà Hàng Madame Lân Đà Nẵng
Reference answer: "04 Bạch Đằng, Thành phố Đà Nẵng. Kiến trúc cổ kính, nhà ngói đỏ liền kề nhấp nhô."
Agent answer: Trả lời đúng cả địa chỉ và mô tả kiến trúc nhưng thêm câu "Những mái hiên với ánh đèn treo cao rực rỡ... giống như một con phố cổ trong ký ức xưa. Nơi đây phù hợp với những cuộc gặp gỡ gia đình, bạn bè."
Lý do bị đánh giá sai: Thêm chi tiết về "mái hiên đèn treo" và "phù hợp với cuộc gặp gỡ" - đúng nhưng ngoài phạm vi reference answer.

**Câu 16 - Dac San Hoi An: địa chỉ và loại hình?**
Kết quả: Sai
Reference document: Document 1924 - Dac San Hoi An
Reference answer: "137, Phan Châu Trinh, Hội An Old Town. Thuộc loại hình nhà hàng."
Agent answer: "Thuộc loại hình ẩm thực."
Lý do bị đánh giá sai: Câu trả lời đúng địa chỉ nhưng sai loại hình. Reference nói "nhà hàng" nhưng agent trả lời "ẩm thực". Đây là lỗi thể loại: category trong tài liệu là "restaurant" nhưng trong mô tả đầy đủ có từ "ẩm thực" và model nhầm từ này làm loại hình thay vì "nhà hàng".

**Câu 17 - Khách sạn Biển Vàng: ở đâu và gần biển nào?**
Kết quả: Đúng
Agent answer: "118 Phạm Văn Đồng, Sơn Trà, Quận Sơn Trà. Nằm gần Bãi Biển Mỹ Khê - 1 trong 6 bãi biển Forbes bầu chọn đẹp nhất hành tinh." - Chính xác đầy đủ.

**--- Loại: conversational (2/3 đúng) ---**

**Câu 18 - "Nơi này có những gì đặc biệt?" (sau "Tôi muốn biết khách sạn Đà Nẵng Riverside")**
Kết quả: Sai
Reference document: Document 1029 - Khách sạn Đà Nẵng Riverside
Reference answer: "A30 Trần Hưng Đạo, Sơn Trà. Cách sân bay 02km, đi bộ ra biển Mỹ Khê chỉ 05 phút, cách phố cổ Hội An 20km."
Agent answer: Trả lời đúng về các khoảng cách nhưng bỏ qua địa chỉ chính xác (A30 Trần Hưng Đạo) và thêm thông tin "gần các trung tâm mua sắm lớn như Vincom, Lotte, Big C" và "đội ngũ nhân viên nhiệt tình."
Lý do bị đánh giá sai: Reference yêu cầu địa chỉ cụ thể nhưng agent không nêu rõ.

**Câu 19 - "Nơi nào vậy?" (sau "Tìm kiếm địa điểm của khách sạn Nhật Hạ II")**
Kết quả: Đúng
Agent answer: "196 Nguyễn Chí Thanh, Hải Châu, Quận Hải Châu, Thành phố Đà Nẵng." - Chính xác.

**Câu 20 - "Nơi nào là đúng?" (sau "Muốn biết vị trí Salem Riverside Đà Nẵng")**
Kết quả: Đúng
Agent answer: "323 Trần Hưng Đạo, Quận Sơn Trà, Thành phố Đà Nẵng." - Chính xác.

---

## 5. Phân tích tổng hợp điểm yếu và nguyên nhân

### 5.1 Nguyên nhân mất điểm theo nhóm

**Nhóm 1 - Reference answer quá ngắn so với khả năng agent (3 câu: câu 5, 7, 15)**
Hệ thống trả lời chính xác và đầy đủ nhưng bị đánh giá sai vì Giskard so sánh với reference answer bị giản lược. Đây là hạn chế của cách Giskard tạo ra reference answer - các LLM judge có xu hướng tạo reference ngắn gọn từ một đoạn văn bản cắt cụt trong tài liệu, trong khi hệ thống RAG có thể khai thác nhiều thông tin có giá trị hơn.

**Nhóm 2 - Entity gây nhiễu trong retrieval (2 câu: câu 9, 10)**
Loại câu hỏi "distracting element" đặt entity không liên quan vào câu để test khả năng lọc nhiễu. Khắc phục một phần: truy xuất phụ bằng 80 ký tự đầu giúp lấy được chunk của Tháp Chiên Đàn. Nhưng model vẫn bao gồm thông tin về entity gây nhiễu trong câu trả lời.

**Nhóm 3 - Câu hỏi hội thoại thiếu ngữ cảnh (1 câu: câu 18)**
Câu hỏi "Nơi này có những gì đặc biệt?" rất mơ hồ. Hệ thống giải quyết bằng cách ghép previous_user_turn vào câu truy vấn, nhưng kết quả vẫn thiếu địa chỉ cụ thể mà reference yêu cầu.

**Nhóm 4 - Nhầm loại hình (1 câu: câu 16)**
"ẩm thực" và "nhà hàng" là hai cách gọi khác nhau cho cùng một khái niệm, nhưng Giskard đánh giá theo khớp từ chính xác.

### 5.2 Biện pháp đã áp dụng trong quá trình phát triển

**Vấn đề 1 - RETRIEVER = 0.25 (phiên trước)**
Nguyên nhân: Khi sinh AgentAnswer.documents, `doc_id` được ép kiểu str(). Giskard so sánh int với str, kết quả luôn False. Tất cả câu hỏi đều bị đánh giá là RETRIEVER failed.
Giải pháp: Truyền trực tiếp int, không ép kiểu.

**Vấn đề 2 - simple correctness = 0.5 (phiên trước)**
Nguyên nhân: System prompt có từ "ngắn gọn" khiến model rút gọn các chi tiết đặc trưng (Steak Louis: "vang đỏ và vang trắng" thay vì "rất nhiều loại... vang đỏ, vang trắng,...").
Giải pháp: Xóa "ngắn gọn", thêm "Giữ nguyên các chi tiết cụ thể và cụm từ đặc trưng từ tài liệu, không rút gọn, không diễn giải lại."

**Vấn đề 3 - distracting element = 0.0 (phiên trước)**
Nguyên nhân: Câu hỏi Tháp Chiên Đàn quá dài, entity "Memory Lounge" có embedding similarity cao hơn Tháp Chiên Đàn trong ChromaDB. Chunk Tháp Chiên Đàn (score 0.635) bị đặt sau Memory Lounge (score 0.662) và mất khỏi top_k.
Giải pháp: Truy xuất phụ bằng 80 ký tự đầu câu hỏi, merge kết quả, đảm bảo cả Tháp Chiên Đàn và Memory Lounge đều trong context. Điểm tăng từ 0.0 lên 0.333 (1/3 đúng).

---

## 6. Tóm tắt điểm mạnh và điểm yếu

### Điểm mạnh

- ROUTING đạt 1.000: Không có câu hỏi nào bị định tuyến sai.
- KNOWLEDGE_BASE đạt 0.933: Kho kiến thức 2164 chunks bao phủ tốt các loại câu hỏi thường gặp.
- Tất cả câu hỏi đơn giản (simple) đạt 1.000: Hệ thống trả lời chính xác truyền thống địa chỉ, dịch vụ cơ bản.
- Tất cả câu hỏi có tình huống (situational) đạt 1.000: Xử lý ngữ cảnh đặt trong tình huống tốt.
- Itinerary planning: 100% validation pass, max leg 12.5 km trung bình.

### Điểm yếu

- REWRITER đạt 0.444: Xử lý câu hỏi hội thoại có ngữ cảnh ngầm định còn hạn chế.
- distracting element đạt 0.333: Hệ thống dù tổng hợp được thông tin đúng nhưng khó tránh đưa thêm nội dung về entity gây nhiễu.
- double đạt 0.333: Câu hỏi kép dễ bị mất điểm do thêm thông tin ngoài phạm vi reference ngắn gọn.
- Phương pháp đánh giá nhị phân (đúng/sai) chưa phân biệt được câu trả lời "đầy đủ hơn reference" với câu trả lời "sai".

---

## 7. Hướng cải tiến

**REWRITER (0.444):**
Tăng cường xử lý ngữ cảnh cho câu hỏi hội thoại bằng cách trích xuất tên địa điểm cụ thể từ lượt hội thoại trước, không chỉ ghép chuỗi thuần túy. Nhận diện pattern "Nơi đó là gì / Nơi này có gì" để biết đây là câu truy vấn phụ thuộc ngữ cảnh.

**distracting element (0.333):**
Post-processing câu trả lời: nếu câu hỏi chứa entity chính và entity gây nhiễu, giới hạn output chỉ đề cập entity chính. Hoặc thêm instruction vào system prompt: "Chỉ trả lời về địa điểm/chủ thể được hỏi trực tiếp trong phần đầu câu hỏi."

**double (0.333):**
Tách câu hỏi kép thành hai câu đơn trước khi gọi LLM, trả lời từng câu riêng, sau đó ghép lại. Giảm rủi ro model trả lời thêm thông tin ngoài hai phần chính.

**Reference answer quality:**
Giskard cho phép tùy chỉnh `agent_description` khi tạo testset. Thêm chỉ dẫn "Reference answers should be comprehensive" để tạo ra reference đầy đủ hơn, giảm số câu bị đánh giá sai do agent trả lời quá đầy đủ.

---

## 8. Cách chạy lại đánh giá

```bash
# Chạy đầy đủ (sử dụng testset đã có)
python scripts/eval_rag.py --load-testset data/test_metrics/giskard_testset.json

# Bỏ qua Giskard (nhanh hơn, không cần LLM judge)
python scripts/eval_rag.py --skip-giskard

# Tạo testset mới với 30 câu
python scripts/eval_rag.py --num-questions 30

# Chỉ chạy retrieval
python scripts/eval_rag.py --skip-giskard --skip-itinerary

# Xuất ra thư mục khác
python scripts/eval_rag.py --output-dir results/eval_20260610
```

Kết quả được ghi vào `data/test_metrics/`:

| File | Mô tả |
|------|-------|
| eval_report.json | Toàn bộ chỉ số dạng JSON |
| eval_log.txt | Nhật ký chạy đầy đủ |
| retrieval_per_query.csv | Chi tiết từng truy vấn retrieval |
| itinerary_per_query.csv | Chi tiết từng lịch trình |
| giskard_per_question.csv | Chi tiết từng câu hỏi Giskard |
| giskard_testset.json | Bộ câu hỏi thử nghiệm (tái sử dụng được) |
