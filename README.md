# Demo phát hiện ngủ gật trong lớp học

Demo Streamlit theo [pipeline.md](pipeline.md), dùng **YOLOv8-Pose + ByteTrack** làm front-end chung với project wall-climbing. MediaPipe chỉ còn dùng cho Face Detection + FaceMesh để giữ nguyên EAR/MAR/head-pose và các luật temporal phía sau.

## Chạy nhanh

Dùng Python **3.10 hoặc 3.11**. Trong dự án này đã có `venv` chứa bộ thư viện phù hợp:

```bash
source venv/bin/activate
python -m pip install -r requirements-classroom.txt
python -m streamlit run classroom_app.py
```

Mở **http://localhost:8501**, chọn video hoặc webcam, rồi nhấn **Bắt đầu infer**. Nút **Dừng và lưu kết quả** kết thúc lượt infer và xuất phần đã xử lý.

Nếu cài mới, tạo môi trường riêng trước:

```bash
python3.10 -m venv .venv-classroom
source .venv-classroom/bin/activate
python -m pip install -r requirements-classroom.txt
python -m streamlit run classroom_app.py
```

Demo giữ `mediapipe==0.10.14` vì sử dụng `mp.solutions` như trong tài liệu. Không dùng môi trường `.venv-yolo11` chưa có MediaPipe/Streamlit. File requirements dành riêng cho demo giúp tái tạo đúng bộ phiên bản, gồm `lapx` cho ByteTrack.

## Đầu vào và kết quả

- **Tải video**: MP4, AVI, MOV, MKV, WEBM, MPEG; giới hạn upload mặc định của Streamlit là 200 MB. Video lớn hơn có thể dùng đường dẫn trên máy, hoặc chạy với `--server.maxUploadSize 1024`.
- **Video trên máy**: nhập đường dẫn, ví dụ `inputs/d_2.mp4`, để thử ngay. Các video tài xế có sẵn chỉ phù hợp kiểm tra module; cần video lớp học để đánh giá nhiều sinh viên thực tế.
- **Webcam trên máy**: camera 0/1/... cắm vào máy chạy Streamlit. Đây là camera phía máy chủ, không phải camera trình duyệt của máy khác. Dùng nút Dừng để giải phóng camera.
- Xem video có box, ID, landmark và bảng EAR/MAR/pitch/trạng thái từng người trong lúc xử lý.
- Video đầu vào có thể xuất MP4 H.264; mọi nguồn đều xuất CSV UTF-8 theo timestamp và ID. Video xuất không có âm thanh. Webcam chỉ xem trực tiếp và xuất CSV.

## Luồng inference

1. **YOLOv8-Pose** phát hiện `person`, trả `bbox + 17 COCO keypoints`; ByteTrack gán ID qua các frame.
2. Từ 17 keypoints, dùng `nose + left/right shoulder` để tính độ gục đầu/vai với đúng công thức cũ.
3. Phát hiện mặt trên crop bằng **MediaPipe Face Detection** mặc định, rồi crop mặt và chạy **FaceMesh**. Nếu có weights **YOLOv8-Face tương thích Ultralytics**, nhập vào sidebar để thay detector mặt. `yolov8n.pt` COCO không phải weights phát hiện mặt.
4. Đo EAR trung bình hai mắt; MAR = khoảng mở môi trong / chiều rộng miệng; pitch bằng `solvePnP`; độ gục đầu/vai = `(y_mũi − y_trung_điểm_vai) / khoảng_cách_hai_vai`.
5. Làm mượt EMA và đếm thời gian liên tục **riêng theo ID**, áp dụng OR-rule theo mức độ: ngủ gục → gật gù → ngáp → tỉnh táo/thiếu dữ liệu.

MediaPipe chạy `static_image_mode=True` để lịch sử landmark của người trước không truyền sang người tiếp theo. Từng lượt infer có model/tracker riêng, không dùng chung tracker giữa các phiên Streamlit. Mất người hoặc mất một tín hiệu sẽ ngắt thời gian liên tục tương ứng.

| Trạng thái | Quy tắc mặc định |
| --- | --- |
| Ngủ gục | EAR < 0.21 **hoặc** pitch đã bù > 25° **hoặc** độ gục đầu/vai > -0.10, liên tục ≥ 2.5 s |
| Gật gù | Nhắm mắt hoặc cúi/gục đầu liên tục ≥ 0.8 s |
| Ngáp | MAR > 0.60 liên tục ≥ 0.8 s |
| Tỉnh táo | Có EAR hợp lệ và chưa có tín hiệu kéo dài vượt ngưỡng |
| Thiếu dữ liệu | Không đủ landmark mắt và chưa đủ bằng chứng từ tư thế |

Ngáp không bắt buộc phải xuất hiện cùng mắt nhắm. Mất mặt đơn thuần không đủ kết luận ngủ gục. CSV có thêm `score` tổng hợp trọng số 0.4 mắt + 0.2 miệng + 0.4 đầu từ thời gian duy trì tín hiệu; đây là chỉ số tham khảo, không phải xác suất. Nhãn dùng OR-rule ở bảng trên.

## Hiệu chỉnh và giới hạn demo

- Quan sát EAR khi mở/nhắm mắt và MAR khi ngáp để đặt ngưỡng phù hợp. Với camera nhìn từ trên xuống, nhập **Pitch tư thế ngồi thẳng** để bù góc camera trước khi xét độ cúi đầu.
- Gật gù ở demo là nhãn theo dấu hiệu kéo dài; chưa xác định chu kỳ gật đầu. Đọc sách, viết bài và nói chuyện có thể tạo dấu hiệu giống ngủ gật/ngáp. Các nhãn là kết quả heuristic, chưa được đánh giá độ chính xác trên lớp học.
- Mặt nhỏ, nghiêng mạnh hoặc che khuất có thể không có landmark. Mặc định bỏ qua mặt có cạnh < 32 pixel trên video đã resize. MediaPipe Face Detection là lựa chọn chạy sẵn; YOLOv8-Face tùy chọn phù hợp để thử cải thiện phát hiện mặt xa theo pipeline.
- Nhiều mặt trong một crop được ghép với vị trí mũi từ pose; nếu không có vị trí mũi và có nhiều mặt, bỏ qua để tránh gán nhầm. Cảnh đông/che khuất vẫn có thể nhầm người hoặc đổi ID; ByteTrack không nhận dạng danh tính.
- MediaPipe chạy CPU; YOLO tự chọn CUDA nếu có. Tốc độ phụ thuộc số người, độ phân giải và thiết bị. Tăng **Infer mỗi N frame** để giảm số lần chạy model. Video xuất lặp lại khung hình vừa infer để giữ thời lượng, nên chuyển động sẽ thưa hơn.
- Ngưỡng thời gian dùng timestamp video (dự phòng chỉ số frame/FPS); webcam dùng thời gian thực. Khoảng trống giữa hai lần quan sát > 1 s sẽ reset bằng chứng liên tục. Video xuất dùng FPS cố định của nguồn; video có FPS biến thiên có thể có thời lượng đầu ra khác.
- Bảng thống kê đếm lần quan sát ở mỗi trạng thái, không phải số đợt ngủ/ngáp. CSV được ghi dần ra file tạm, sau đó trả về để tải; file tạm được dọn khi hoàn tất, dừng hoặc lỗi. Video kết quả được giữ trong phiên Streamlit để tải xuống.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Kiểm tra gồm nhắm mắt/ngáp/gục đầu độc lập, mất landmark, thời gian theo FPS, tách lịch sử từng người, quy ước góc pitch và xuất video giữ số frame. Các test logic dùng tín hiệu tổng hợp, không thay thế đánh giá với nhãn lớp học.

Đã chạy 16 test tự động và kiểm tra Streamlit với video thật, bao gồm hoàn tất, dừng giữa chừng, tải MP4/CSV và thông báo thiếu weights. Clip thử ghép hai video tài xế giữ đúng tỷ lệ ảnh cho 2 ID và 12/12 lượt quan sát có landmark mặt. Kết quả thử trong workspace: `outputs/classroom_smoke.mp4`, `outputs/classroom_smoke.csv`. Chưa thử webcam vật lý hoặc đánh giá độ chính xác trên video lớp học có nhãn.

## Các file được giữ trong Git

- [classroom_app.py](classroom_app.py): giao diện Streamlit.
- [classroom_core.py](classroom_core.py): YOLOv8-Pose, ByteTrack, MediaPipe FaceMesh và luật phân loại theo thời gian.
- [pose_tracking.py](pose_tracking.py): front-end chung `YOLOv8-Pose + ByteTrack -> bbox + ID + 17 keypoints`.
- [pose_bytetrack.yaml](pose_bytetrack.yaml): cấu hình tracking dùng chung.
- [requirements-classroom.txt](requirements-classroom.txt): thư viện để chạy demo và kiểm thử.
- `yolov8l-pose.pt`: pretrained YOLOv8-Pose dùng chung với project wall-climbing. Nếu file chưa có, Ultralytics sẽ thử tải official weight ở lần chạy đầu; khi chạy offline, copy file `yolov8l-pose.pt` từ repo `samoonz/wall-climbing-intrusion-detection` vào thư mục gốc.
- [tests/test_classroom_core.py](tests/test_classroom_core.py): kiểm thử logic, hình học và xuất video.
- `README.md`, [pipeline.md](pipeline.md), [report.md](report.md): hướng dẫn và báo cáo phương pháp hiện tại.
- [.gitignore](.gitignore): chỉ cho phép Git theo dõi các file trong danh sách này.

Mã demo cũ, dataset, kết quả train, video, kết quả inference và môi trường Python được bỏ qua. Khi thêm file cần thiết cho phương pháp hiện tại, bổ sung ngoại lệ tương ứng trong `.gitignore`.

Tham khảo API: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [MediaPipe FaceMesh phiên bản 0.10.14](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/docs/solutions/face_mesh.md), [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment).


## Front-end dùng chung với wall-climbing

Hai project dùng cùng contract đầu vào:

```text
YOLOv8-Pose + ByteTrack
        ↓
bbox + track ID + 17 COCO keypoints
```

Trong project drowsiness, 17 keypoints chỉ thay nguồn MediaPipe Pose cho phần posture. Face Detection, FaceMesh, EAR, MAR, solvePnP pitch, EMA, timer và luật Awake/Yawn/Nod/Sleep được giữ nguyên.
