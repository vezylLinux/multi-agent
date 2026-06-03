# Vấn đề Đánh giá Hệ thống Lập kế hoạch Du lịch Đa tác nhân

## 1. Bối cảnh

Hệ thống hiện tại đã triển khai một số cơ chế tự kiểm tra nội bộ:

- **Structural validation** (validator_node): kiểm tra số ngày, cấu trúc slot, số hoạt động/ngày
- **Distance validation**: phát hiện leg di chuyển bất hợp lý (>18 km, >25 km)
- **Retry mechanism**: tự tái lập kế hoạch khi phát hiện vi phạm

Tuy nhiên, toàn bộ các cơ chế trên là **self-reported**  hệ thống tự đánh giá chính mình. Không có nguồn sự thật độc lập (ground truth) nào để so sánh. Đây là rào cản cốt lõi khiến việc xây dựng bộ metrics đánh giá khách quan trở nên đặc biệt khó khăn.

---

## 2. Phân tích các thách thức chính

### 2.1 Không có Ground Truth

Bài toán phân loại hay dịch máy có nhãn đúng/sai xác định. Lịch trình du lịch thì không.

Câu hỏi "lịch trình nào tốt hơn?" không có đáp án khách quan. Cùng một query:

> *"3 ngày Đà Nẵng, thích biển và hải sản, ngân sách tiết kiệm"*

Hai lịch trình khác nhau hoàn toàn đều có thể "đúng":
- Lịch trình A tập trung vào Mỹ Khê + Sơn Trà
- Lịch trình B tập trung vào Ngũ Hành Sơn + Hải Châu

Không có cơ sở nào để khẳng định cái nào tốt hơn mà không có phản hồi từ người dùng thực tế.

### 2.2 Tính Đa chiều của Output

Một lịch trình tốt phải thỏa mãn đồng thời nhiều tiêu chí không đồng nhất:

| Chiều đánh giá | Cách đo | Vấn đề |
|---|---|---|
| Tính đúng đắn về nội dung | Địa điểm có tồn tại thực không? | DB có thể lỗi thời |
| Khả năng thực hiện | Khoảng cách di chuyển có hợp lý? |  Đã đo được (TrackAsia) |
| Phù hợp sở thích | Có đúng interest tag không? | Mapping mờ, không tuyệt đối |
| Phù hợp ngân sách | Khách sạn/nhà hàng có đúng giá không? | DB không có dữ liệu giá |
| Chất lượng ngôn ngữ | Văn phong tiếng Việt tự nhiên không? | Cần human evaluation |
| Tính đa dạng | Không lặp lại loại hình địa điểm | Có thể đo, nhưng không rõ ngưỡng |
| Trải nghiệm cảm xúc | Lịch trình có "cảm giác" hay không? | Hoàn toàn chủ quan |

Không có metric đơn lẻ nào bao quát được tất cả. Tổng hợp các metric thành một điểm số duy nhất đòi hỏi trọng số  mà trọng số đó lại mang tính tùy tiện.

### 2.3 Tính Chủ quan và Phụ thuộc Ngữ cảnh

Cùng một lịch trình có thể:
- Hoàn hảo với người thích di chuyển nhiều, khám phá xa
- Quá mệt với gia đình có trẻ nhỏ
- Quá nhàm với khách đã đến Đà Nẵng nhiều lần

Ngay cả khi thu thập đánh giá từ người dùng thực, điểm số sẽ bị confounded bởi:
- Sở thích cá nhân
- Kinh nghiệm du lịch trước đó
- Thời tiết thực tế trong chuyến đi
- Sự kiện ngoài tầm kiểm soát (địa điểm đóng cửa, đông người)

### 2.4 LLM Non-determinism

Planner sử dụng `temperature=0.3`. Cùng một query chạy 10 lần cho 10 lịch trình khác nhau  tất cả đều "pass validation". Điều này đặt ra câu hỏi:

- Metrics nào nên đánh giá trên **run nào**?
- Có nên lấy **trung bình** qua nhiều lần chạy? (tốn kém, chậm)
- **Variance** cao hay thấp thì tốt hơn? (đa dạng vs nhất quán)

Không có thỏa thuận chuẩn nào cho bài toán này trong cộng đồng nghiên cứu hiện tại.

### 2.5 Lỗi Lan Truyền qua Pipeline Đa tác nhân

Hệ thống có 5 tác nhân nối tiếp. Lỗi từ upstream lan xuống downstream mà không để lại dấu vết rõ ràng:

```
intake_node  →  retrieval_node  →  planning_node  →  validator_node  →  response_node
   ↓                ↓                   ↓                  ↓
Wrong intent?   Wrong pool?        Bad plan?          False positive?
```

Ví dụ: nếu **intake** trích xuất sai `interests` → **retrieval** trả về pool sai → **planning** tạo lịch trình sai → **validator** pass vì cấu trúc đúng → người dùng nhận lịch trình không phù hợp sở thích.

Không thể quy trách nhiệm cho đúng tác nhân gây lỗi nếu chỉ đánh giá đầu ra cuối. Cần **per-agent evaluation** — tốn kém gấp 5 lần và mỗi tác nhân lại có ground truth khác nhau.

### 2.6 Thiếu Dữ liệu Phản hồi Người dùng

Hệ thống không có:
- Cơ chế rating (👍/👎) sau mỗi lịch trình
- Dữ liệu người dùng thực sự đi theo lịch trình hay không
- A/B testing giữa các phiên bản
- Click-through rate trên từng địa điểm được gợi ý
- Follow-up sau chuyến đi

Không có vòng phản hồi này, mọi cải thiện thuật toán đều không thể xác nhận có thực sự cải thiện trải nghiệm người dùng hay không.

## 3. Tại sao không thể dùng LLM-as-judge?

Cách tiếp cận phổ biến trong nghiên cứu là dùng một LLM mạnh hơn (như GPT-4) để đánh giá output của LLM yếu hơn. Tuy nhiên với bài toán này:

1. **Bias tự đánh giá**: GPT-4 cũng là OpenAI model, có thể ưu tiên các pattern nó đã học
2. **Không có ngữ cảnh địa phương**: LLM không biết "Hải sản Năm Đảnh" là nhà hàng nổi tiếng Đà Nẵng
3. **Không xác minh được địa điểm thực tế**: LLM không gọi TrackAsia để verify địa chỉ
4. **Tốn kém**: N lịch trình × M tiêu chí × LLM call = chi phí cao
5. **Vẫn không có ground truth**: LLM judge chỉ thay thế human judge, không phải đánh giá khách quan

---

## 4. Hướng tiếp cận khả thi

Thay vì đánh giá tổng thể — không khả thi với nguồn lực hiện tại — có thể đánh giá từng khía cạnh độc lập:

| Khía cạnh | Metric khả thi | Cách triển khai |
|---|---|---|
| **Tính thực thi** | % legs có distance hợp lý (<18km) |  Đã có — TrackAsia |
| **Độ phủ intent** | % interests được map vào plan | Tính được từ INTEREST_KEYWORDS |
| **Độ chính xác geocode** | % địa điểm có lat/lon hợp lệ |  99.9% hiện tại |
| **Consistency** | Variance kết quả qua N lần chạy cùng query | Chạy N=10, đo std(max_leg_km) |
| **Budget alignment** | Hostel khi `low`, resort khi `high` | Kiểm tra được từ hotel.name + star |
| **Structural pass rate** | % plans pass validator |  Đã track trong logs |
| **Response latency** | Trung bình và P95 response time |  Đã có trong timings |

**Điều không thể thay thế**: thu thập feedback người dùng thực — dù chỉ là một nút "Lịch trình này có hữu ích không?" — là bước cần thiết nhất để đánh giá chất lượng thực sự của hệ thống.

---
