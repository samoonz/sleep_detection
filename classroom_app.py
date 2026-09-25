"""Run: python -m streamlit run classroom_app.py"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import streamlit as st

from classroom_core import ClassroomSettings, InferenceRun, LABELS, ROOT

st.set_page_config(page_title="Phát hiện ngủ gật trong lớp học", page_icon="🎓", layout="wide")


def show_observations(rows):
    if not rows:
        st.info("Chưa có người được gán ID trong khung hình này.")
        return
    st.dataframe([
        {"ID": row["id"], "Trạng thái": row["label"],
         "EAR": row["ear"], "MAR": row["mar"], "Pitch (°)": row["pitch"],
         "Độ gục đầu/vai": row["head_drop"], "Nhắm mắt (s)": round(row["eye_seconds"], 2),
         "Gục đầu (s)": round(row["head_seconds"], 2), "Ngáp (s)": round(row["yawn_seconds"], 2),
         "Thấy mặt": row["face_visible"], "Lý do": row["reason"]}
        for row in rows
    ], use_container_width=True, hide_index=True)


def show_result(result):
    st.subheader("Kết quả lần infer gần nhất")
    st.caption("Đã dừng theo yêu cầu." if result["stopped"] else "Đã xử lý xong nguồn đầu vào.")
    columns = st.columns(4)
    columns[0].metric("Số ID đã theo dõi", len(result["people"]))
    columns[1].metric("Khung hình đã đọc", result["frames"])
    columns[2].metric("Tốc độ inference", f'{result["inference_fps"]:.1f} FPS')
    columns[3].metric("Thời gian xử lý", f'{result["elapsed"]:.1f} s')
    st.caption("FPS inference tính trên các khung hình chạy model, không gồm thời gian giao diện và mã hóa video.")
    if result["fps_fallback"]:
        st.warning("Nguồn không cung cấp FPS hợp lệ; demo dùng 25 FPS dự phòng. Hãy kiểm tra thời gian trong CSV.")
    if result["people"]:
        st.dataframe([
            {"ID": person["id"], "Lần quan sát": person["observations"],
             "Thấy mặt (%)": round(100 * person["face_observations"] / person["observations"], 1),
             **{label: person[state] for state, label in LABELS.items()}}
            for person in result["people"]
        ], use_container_width=True, hide_index=True)
        st.caption("Các cột trạng thái đếm số lần quan sát, không phải số sự kiện. ID có thể đổi khi người bị che khuất lâu.")
    else:
        st.info("Không phát hiện được người để phân tích. Thử giảm confidence hoặc dùng video rõ người hơn.")
    if result["video"]:
        if result["browser_video"]:
            st.video(result["video"])
        else:
            st.warning("Không mã hóa được H.264. Bạn có thể tải MP4 gốc và mở bằng VLC.")
        st.download_button("Tải video đã gắn nhãn", result["video"], "classroom_annotated.mp4", "video/mp4")
    st.download_button("Tải CSV theo từng ID và thời điểm", result["csv"], "classroom_observations.csv", "text/csv")
    with st.expander("Cấu hình của kết quả này"):
        st.json(result["settings"])


st.title("🎓 Phát hiện ngủ gật trong lớp học")
st.caption("YOLOv8-Pose → ByteTrack → bbox + ID + 17 keypoints → FaceMesh → trạng thái theo thời gian")
active = st.session_state.get("classroom_run") is not None

with st.sidebar:
    st.header("Cấu hình demo")
    st.caption("Cấu hình được giữ cố định trong mỗi lần infer.")
    with st.expander("Model và hiệu năng", expanded=True):
        person_weights = st.text_input("Weights YOLOv8-Pose", str(ROOT / "yolov8l-pose.pt"), disabled=active)
        face_weights = st.text_input("Weights YOLOv8-Face (tùy chọn)", disabled=active,
                                     help="Để trống: dùng MediaPipe Face Detection trên crop người.")
        device = st.selectbox("Thiết bị YOLO", ["auto", "cpu", "cuda:0"], disabled=active)
        confidence = st.slider("Confidence người", 0.10, 0.90, 0.35, 0.05, disabled=active)
        stride = st.slider("Infer mỗi N frame", 1, 6, 2, disabled=active)
        max_people = st.slider("Số người tối đa mỗi frame", 1, 50, 20, disabled=active)
        image_size = st.selectbox("Kích thước YOLO", [640, 960, 1280], disabled=active)
        max_width = st.selectbox("Chiều rộng video xử lý tối đa", [1280, 1920, 640, 960], disabled=active)
        draw_landmarks = st.checkbox("Vẽ landmark mắt, miệng và tư thế", True, disabled=active)
    with st.expander("Ngưỡng mắt, miệng và đầu"):
        ear = st.slider("EAR dưới ngưỡng → mắt nhắm", 0.10, 0.40, 0.21, 0.01, disabled=active)
        mar = st.slider("MAR trên ngưỡng → mở miệng", 0.20, 1.20, 0.60, 0.05, disabled=active)
        pitch = st.slider("Góc cúi đầu (°)", 10.0, 60.0, 25.0, 1.0, disabled=active)
        pitch_offset = st.slider("Pitch tư thế ngồi thẳng (°)", -40.0, 40.0, 0.0, 1.0, disabled=active,
                                 help="Đọc Pitch khi ngồi thẳng rồi nhập tại đây để bù góc camera.")
        head_drop = st.slider("Ngưỡng độ gục đầu/vai", -0.60, 0.50, -0.10, 0.05, disabled=active,
                              help="(y_mũi − y_trung điểm vai) / khoảng cách hai vai. Giá trị tăng khi đầu xuống thấp.")
        min_face = st.slider("Cạnh mặt tối thiểu (pixel)", 16, 100, 32, 4, disabled=active)
    with st.expander("Ngưỡng thời gian"):
        nod = st.slider("Nhắm mắt/cúi đầu → gật gù (s)", 0.3, 2.0, 0.8, 0.1, disabled=active)
        sleep = st.slider("Nhắm mắt/gục đầu → ngủ gục (s)", 2.0, 6.0, 2.5, 0.1, disabled=active)
        yawn = st.slider("Mở miệng → ngáp (s)", 0.3, 3.0, 0.8, 0.1, disabled=active)
    st.info("Tín hiệu mắt và gục đầu được xét độc lập. Mất mặt đơn thuần được ghi là thiếu dữ liệu.")

settings = ClassroomSettings(
    person_weights=person_weights, face_weights=face_weights.strip(), device=device,
    confidence=confidence, stride=stride, max_people=max_people, image_size=image_size,
    max_width=max_width, draw_landmarks=draw_landmarks, ear_threshold=ear,
    mar_threshold=mar, pitch_threshold=pitch, pitch_offset=pitch_offset,
    head_drop_threshold=head_drop, min_face_pixels=min_face,
    nod_seconds=nod, sleep_seconds=sleep, yawn_seconds=yawn,
)

source_kind = st.radio("Nguồn đầu vào", ["Tải video", "Video trên máy", "Webcam trên máy"], horizontal=True, disabled=active)
source = None
suffix = ".mp4"
source_key = source_kind
if source_kind == "Tải video":
    upload = st.file_uploader("Chọn video lớp học", type=["mp4", "avi", "mov", "mkv", "webm", "mpeg", "mpg"], disabled=active)
    if upload:
        source = upload.getvalue()
        suffix = Path(upload.name).suffix.lower()
        source_key += hashlib.sha256(source).hexdigest()
        if not active:
            with st.expander("Xem video đầu vào"):
                st.video(source)
elif source_kind == "Video trên máy":
    source = st.text_input("Đường dẫn video trên máy chạy Streamlit", disabled=active)
    source_key += source
    if source:
        source = str(Path(source).expanduser())
        if not Path(source).is_file():
            st.warning("Chưa tìm thấy video tại đường dẫn này.")
            source = None
else:
    source = int(st.number_input("Chỉ số webcam", min_value=0, max_value=10, value=0, disabled=active))
    source_key += str(source)
    st.info("Sử dụng webcam cắm vào máy chạy Streamlit. Đóng ứng dụng khác đang dùng camera. Chế độ này xuất CSV và hiển thị trực tiếp.")

if st.session_state.get("classroom_source") != source_key and not active:
    st.session_state["classroom_source"] = source_key
    st.session_state.pop("classroom_result", None)

if st.button("▶ Bắt đầu infer", type="primary", disabled=active or source is None or source == ""):
    st.session_state.pop("classroom_result", None)
    try:
        with st.spinner("Đang nạp YOLOv8-Pose, ByteTrack và MediaPipe Face..."):
            st.session_state["classroom_run"] = InferenceRun(source, settings, suffix)
        st.rerun()
    except Exception as exc:
        st.error(f"Không khởi động được inference: {exc}")


@st.fragment(run_every=0.2 if active else None)
def inference_panel():
    run = st.session_state.get("classroom_run")
    if run is None:
        if st.session_state.get("classroom_result") is not None:
            show_result(st.session_state["classroom_result"])
        return
    stop = st.button("⏹ Dừng và lưu kết quả", type="secondary")
    done = stop
    try:
        # Return control regularly so Streamlit can handle Stop and rerenders.
        deadline = time.perf_counter() + 0.25
        while not done and time.perf_counter() < deadline:
            done = not run.step()
        if run.preview is not None:
            st.image(run.preview, channels="BGR", use_column_width=True)
        if run.total_frames:
            st.progress(min(run.frame_count / run.total_frames, 1.0), text=f"Đã đọc {run.frame_count}/{run.total_frames} frame")
        else:
            st.caption(f"Đã đọc {run.frame_count} frame · {run.last_timestamp:.1f} s")
        show_observations(run.rows)
        if done:
            with st.spinner("Đang lưu CSV và video kết quả..."):
                st.session_state["classroom_result"] = run.finish(stopped=stop)
            st.session_state.pop("classroom_run", None)
            st.rerun()
    except Exception as exc:
        run.close()
        st.session_state.pop("classroom_run", None)
        st.session_state["classroom_error"] = str(exc)
        st.rerun()


if "classroom_error" in st.session_state:
    st.error(f'Inference gặp lỗi: {st.session_state.pop("classroom_error")}')
inference_panel()
st.caption("Các ngưỡng là cấu hình demo cần hiệu chỉnh theo video lớp học. Khi bỏ frame, video xuất lặp lại khung hình vừa infer và không có âm thanh.")
