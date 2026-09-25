> **Cập nhật triển khai:** code hiện tại đã thay MediaPipe Pose bằng YOLOv8-Pose + ByteTrack để dùng chung front-end với wall-climbing. Đầu ra chung là bbox + ID + 17 COCO keypoints. MediaPipe Face Detection/FaceMesh và toàn bộ logic EAR/MAR/pitch/temporal phía sau vẫn giữ nguyên.

# Hướng dẫn xây dựng hệ thống phát hiện sinh viên ngủ gật trong lớp học
### (Pipeline 2 giai đoạn — chỉ dùng pretrained model)

---

## 1. Tổng quan kiến trúc

```
Camera lớp học
      │
      ▼
[Giai đoạn 1] Person + Face Detection
      │  → bounding box từng sinh viên
      │  → bounding box khuôn mặt (nếu có)
      ▼
[Giai đoạn 2] Pose Estimation + Face Landmark (per-person, crop theo box GĐ1)
      │  → 17/33 keypoint cơ thể
      │  → 468 landmark khuôn mặt (hoặc 68 nếu dùng dlib)
      ▼
[Logic phân loại trạng thái]
      │  → EAR (mắt), MAR (miệng), góc nghiêng đầu, độ gục vai
      ▼
Trạng thái: Tỉnh táo / Gật gù / Ngủ gục / Ngáp
```

So với bài toán drowsy-driving bạn đang làm (1 tài xế, camera cố định gần mặt), bài toán lớp học khó hơn ở chỗ: **nhiều người trong khung hình, góc camera xa, khuôn mặt nhỏ, dễ bị che khuất**. Vì vậy kiến trúc 2 giai đoạn (detect người trước, crop rồi mới phân tích chi tiết) là lựa chọn đúng — làm thẳng landmark trên ảnh toàn cảnh sẽ rất kém chính xác.

---

## 2. Giai đoạn 1: Phát hiện người và khuôn mặt

### 2.1. Phát hiện người (Person Detection)

| Model | Nguồn pretrained | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **YOLOv8n/s (COCO)** | `ultralytics` (pip) | Nhanh, nhẹ, class "person" có sẵn, dễ chạy realtime CPU | Box người không kèm landmark |
| **YOLOv10 / YOLO11 (COCO)** | `ultralytics` | Chính xác hơn YOLOv8 cùng cỡ, tốc độ tương đương | Mới hơn, ít tài liệu troubleshoot |
| **RT-DETR** | `ultralytics` | Không cần NMS, chính xác cao | Chậm hơn YOLO trên CPU |

**Khuyến nghị**: dùng **YOLOv8n hoặc YOLOv8s** (bạn đã quen với YOLOv8 từ dự án drowsy driving) với `model.predict(classes=[0])` để chỉ lấy class "person". Đây là model bạn có thể tái sử dụng kinh nghiệm sẵn có.

### 2.2. Phát hiện khuôn mặt (Face Detection)

Vì lớp học chụp xa, mặt nhỏ → cần model face detector chuyên biệt, không nên dùng lại face detection tổng quát của MediaPipe Holistic (được tối ưu cho mặt lớn, gần camera).

| Model | Nguồn pretrained | Ghi chú |
|---|---|---|
| **YOLOv8-Face** (repo `derronqi/yolov8-face` hoặc các bản fork trên HuggingFace) | weights `.pt` có sẵn | Phát hiện mặt nhỏ tốt hơn MediaPipe, cùng framework với YOLOv8 person → dễ tích hợp |
| **RetinaFace** (`insightface`) | pip `insightface`, weight `retinaface_r50_v1` | Rất mạnh với mặt nhỏ/nghiêng, có kèm 5 điểm landmark thô | Nặng hơn, cần GPU để realtime tốt |
| **SCRFD** (`insightface`) | pip `insightface` | Nhẹ hơn RetinaFace, vẫn tốt với mặt nhỏ, phù hợp CPU | |
| **MediaPipe Face Detection (BlazeFace)** | `mediapipe` pip | Rất nhanh, nhẹ | Yếu với mặt nhỏ ở xa camera — cần test kỹ với dữ liệu lớp học thật |

**Khuyến nghị thực tế**: chạy **YOLOv8-Face hoặc SCRFD** trên **crop từ bounding box người ở GĐ1** (không chạy trên toàn ảnh) — vừa tăng độ chính xác (mặt lúc này chiếm tỉ lệ lớn hơn trong crop), vừa giảm compute vì chỉ xử lý vùng ROI.

---

## 3. Giai đoạn 2: Pose + Face Landmark để xác định trạng thái

### 3.1. Pose Estimation (keypoint cơ thể)

| Model | Nguồn | Keypoint | Ghi chú |
|---|---|---|---|
| **MediaPipe Pose (BlazePose)** | `mediapipe` pip | 33 điểm | Nhẹ, chạy tốt CPU, có landmark đầu-vai-cổ chi tiết → rất hợp để đo góc gục đầu/vai |
| **YOLOv8-Pose** | `ultralytics` | 17 điểm (COCO format) | Cùng framework với GĐ1, dễ tích hợp pipeline, nhưng ít điểm vùng đầu/cổ hơn BlazePose |
| **MoveNet (Lightning/Thunder)** | TensorFlow Hub | 17 điểm | Rất nhanh, tối ưu cho thiết bị yếu, nhưng cần TF runtime riêng |

**Khuyến nghị**: **MediaPipe Pose** — vì bạn đã dùng MediaPipe cho FaceMesh trong dự án drowsy driving, giữ chung 1 framework sẽ giảm effort tích hợp, và BlazePose có điểm `nose`, `left/right_ear`, `left/right_shoulder` đủ để tính góc nghiêng đầu so với trục vai (chỉ số quan trọng để phát hiện "gục đầu xuống bàn").

### 3.2. Face Landmark

| Model | Nguồn | Điểm | Dùng để tính |
|---|---|---|---|
| **MediaPipe FaceMesh** | `mediapipe` pip | 468 điểm | EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), head pose 3D (pitch/yaw/roll qua `solvePnP`) |
| **dlib 68-point** | `dlib` pretrained `shape_predictor_68_face_landmarks.dat` | 68 điểm | EAR/MAR cổ điển, nhẹ hơn nhưng kém chính xác với mặt nghiêng/xa |

**Khuyến nghị**: giữ **MediaPipe FaceMesh** (bạn đang dùng rồi) vì 468 điểm cho phép tính head pose 3D chính xác hơn — quan trọng khi sinh viên cúi gằm mặt xuống, góc camera lớp học thường chếch từ trên xuống.

---

## 4. Logic phân loại trạng thái ngủ gật

Đây là phần bạn từng gặp vấn đề recall thấp với rule AND cứng trong dự án driving. Với bối cảnh lớp học, mình đề xuất cải tiến theo 3 hướng:

### 4.1. Các chỉ số đầu vào (per-person, per-frame)

- **EAR (Eye Aspect Ratio)**: mắt nhắm lâu → buồn ngủ
- **MAR (Mouth Aspect Ratio)**: ngáp
- **Head pitch angle**: đầu cúi xuống quá ngưỡng (từ landmark mặt hoặc từ vector nose–shoulder trong pose)
- **Shoulder-head displacement**: đầu gục xuống thấp hơn vai đáng kể (dấu hiệu ngủ gục trên bàn — dùng riêng pose, không cần mặt nếu mặt bị che)

### 4.2. Cải tiến thay vì AND-rule cứng

Thay vì `EAR thấp AND MAR cao AND góc đầu lớn` (dễ miss recall như bạn từng gặp), dùng:

- **Weighted scoring**: mỗi chỉ số góp một điểm số theo trọng số, tổng vượt ngưỡng → cảnh báo. Ví dụ: `score = 0.4*eye_score + 0.2*yawn_score + 0.4*head_pose_score`
- **OR-rule có phân cấp mức độ nghiêm trọng**: EAR thấp kéo dài (>N frame) → "gật gù"; head_pitch vượt ngưỡng lớn kéo dài → "ngủ gục" (độc lập, không cần EAR vì mắt có thể bị che khi gục đầu)
- **Temporal smoothing**: dùng sliding window hoặc EMA (exponential moving average) trên vài frame liên tiếp thay vì đánh giá từng frame đơn lẻ — giảm nhiễu do chớp mắt bình thường, giảm false positive/negative

### 4.3. Nhãn trạng thái đề xuất

| Trạng thái | Điều kiện chính |
|---|---|
| Tỉnh táo | EAR bình thường, head pitch nhỏ |
| Ngáp | MAR cao, thời gian ngắn |
| Gật gù (nodding) | EAR thấp theo chu kỳ, head pitch dao động |
| Ngủ gục (sleeping) | Head pitch lớn kéo dài liên tục HOẶC mặt bị che bởi tay/bàn kéo dài (face detector GĐ1 mất mặt liên tục nhiều frame là tín hiệu bổ sung) |

---

## 5. Pipeline tích hợp — khung code tham khảo

```python
from ultralytics import YOLO
import mediapipe as mp
import cv2

# Giai đoạn 1
person_model = YOLO("yolov8n.pt")          # pretrained COCO, class 0 = person
face_model = YOLO("yolov8n-face.pt")       # pretrained face detector

# Giai đoạn 2
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

def process_frame(frame):
    results = []
    persons = person_model.predict(frame, classes=[0], verbose=False)[0]

    for box in persons.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        crop = frame[y1:y2, x1:x2]

        # Pose trên crop người
        pose_res = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Face detect trên crop người (thu hẹp vùng tìm mặt)
        face_res = face_model.predict(crop, verbose=False)[0]

        face_landmarks = None
        if len(face_res.boxes) > 0:
            fx1, fy1, fx2, fy2 = face_res.boxes.xyxy[0].cpu().numpy().astype(int)
            face_crop = crop[fy1:fy2, fx1:fx2]
            mesh_res = mp_face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            if mesh_res.multi_face_landmarks:
                face_landmarks = mesh_res.multi_face_landmarks[0]

        state = classify_state(pose_res, face_landmarks)  # logic mục 4
        results.append({"box": (x1, y1, x2, y2), "state": state})

    return results
```

> Lưu ý: nên cache/track ID từng sinh viên qua các frame (ví dụ dùng ByteTrack có sẵn trong `ultralytics` qua `model.track()`) để tính temporal smoothing theo từng người, thay vì xử lý độc lập từng frame.

---

## 6. Dataset để test/đánh giá (không cần train, chỉ để validate pipeline)

- **SUST Driver Drowsiness Dataset** — bạn đã dùng, có thể dùng để test riêng module EAR/MAR trước khi ghép vào bối cảnh lớp học
- **DROZY dataset** — video drowsiness có nhãn mức độ buồn ngủ (Karolinska Sleepiness Scale)
- **NTHU Drowsy Driver Detection Dataset** — nhiều điều kiện ánh sáng, đeo kính, có ngáp
- Với bối cảnh lớp học cụ thể: nên **tự quay/thu thập một tập nhỏ video lớp học thật** (vài chục clip, có label thủ công) để tinh chỉnh ngưỡng (threshold) — vì góc camera, khoảng cách, độ sáng lớp học khác hẳn dataset driving.

---

## 7. Thư viện cần cài

```bash
pip install ultralytics mediapipe opencv-python numpy
# Nếu dùng RetinaFace/SCRFD:
pip install insightface onnxruntime
```

---

## 8. Hướng mở rộng sau POC

- **Multi-face tracking**: gắn ID ổn định cho từng sinh viên qua thời gian (ByteTrack/DeepSORT) để thống kê "sinh viên X ngủ gật bao nhiêu phút trong buổi học"
- **Alert/dashboard**: xuất log CSV theo thời gian thực + timestamp, hoặc cảnh báo tức thời cho giảng viên
- **Xử lý che khuất**: khi tay/bàn che mặt hoàn toàn nhiều frame liên tục — dùng tín hiệu pose (đầu gục xuống, không thấy mặt) làm proxy cho "ngủ gục" thay vì phụ thuộc hoàn toàn vào face landmark
- **Cân nhắc đạo đức/quyền riêng tư**: tương tự phần bạn đã chuẩn bị cho tài liệu POC drowsy driving — cần có phần thông báo/đồng thuận nếu triển khai giám sát sinh viên thực tế