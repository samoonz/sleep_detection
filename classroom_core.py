"""Per-student YOLOv8-Pose + ByteTrack + MediaPipe Face inference.

The shared front-end returns bbox + track ID + 17 COCO keypoints for every
person. MediaPipe is kept only for face detection / FaceMesh so EAR, MAR,
head-pose and all downstream temporal rules remain unchanged.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/classroom-matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/classroom-ultralytics")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np

from pose_tracking import track_people

ROOT = Path(__file__).resolve().parent
LABELS = {
    "awake": "Tỉnh táo", "yawning": "Ngáp", "nodding": "Gật gù",
    "sleeping": "Ngủ gục", "unknown": "Thiếu dữ liệu",
}
COLORS = {
    "awake": (80, 190, 70), "yawning": (0, 200, 255),
    "nodding": (0, 130, 255), "sleeping": (40, 40, 230),
    "unknown": (160, 160, 160),
}
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
PNP_IDS = (1, 152, 33, 263, 61, 291)
# Camera convention: x right, y down, z away; positive pitch means looking down.
FACE_MODEL = np.array([
    (0, 0, 0), (0, 63.6, 12.5), (-43.3, -32.7, 26.0),
    (43.3, -32.7, 26.0), (-28.9, 28.9, 24.1), (28.9, 28.9, 24.1),
], dtype=np.float64)


@dataclass(frozen=True)
class ClassroomSettings:
    person_weights: str = str(ROOT / "yolov8l-pose.pt")
    face_weights: str = ""
    device: str = "auto"
    confidence: float = 0.35
    image_size: int = 640
    max_people: int = 20
    stride: int = 2
    max_width: int = 1280
    min_face_pixels: int = 32
    ear_threshold: float = 0.21
    mar_threshold: float = 0.60
    pitch_threshold: float = 25.0
    pitch_offset: float = 0.0
    head_drop_threshold: float = -0.10
    nod_seconds: float = 0.8
    sleep_seconds: float = 2.5
    yawn_seconds: float = 0.8
    smoothing_seconds: float = 0.15
    max_gap_seconds: float = 1.0
    draw_landmarks: bool = True

    def __post_init__(self):
        if not 0 < self.confidence < 1:
            raise ValueError("Confidence phải nằm trong (0, 1).")
        if min(self.stride, self.max_people, self.min_face_pixels) < 1:
            raise ValueError("Stride, số người và kích thước mặt phải dương.")
        if self.sleep_seconds < self.nod_seconds or self.nod_seconds <= 0:
            raise ValueError("Thời gian ngủ gục phải >= thời gian gật gù > 0.")
        if min(self.yawn_seconds, self.max_gap_seconds) <= 0 or self.smoothing_seconds < 0:
            raise ValueError("Cấu hình thời gian không hợp lệ.")


@dataclass
class Signals:
    ear: float | None = None
    mar: float | None = None
    pitch: float | None = None
    head_drop: float | None = None
    face_visible: bool = False
    pose_visible: bool = False


class StudentState:
    """Measure continuous evidence in source seconds, never inference wall time."""

    def __init__(self):
        self.last_timestamp = None
        self.filtered = {}
        self.since = {name: None for name in ("eye", "mouth", "head")}
        self.durations = {name: 0.0 for name in self.since}

    def update(self, signals: Signals, timestamp: float, cfg: ClassroomSettings) -> dict:
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp không hợp lệ.")
        dt = 0.0 if self.last_timestamp is None else timestamp - self.last_timestamp
        if dt < 0 or dt > cfg.max_gap_seconds:
            self.filtered.clear()
            self.since = dict.fromkeys(self.since)
        self.last_timestamp = timestamp
        alpha = 1.0 if cfg.smoothing_seconds == 0 else 1 - math.exp(-max(dt, 0) / cfg.smoothing_seconds)
        smooth = {}
        for name in ("ear", "mar", "pitch", "head_drop"):
            value = getattr(signals, name)
            if value is None or not math.isfinite(value):
                self.filtered.pop(name, None)
                smooth[name] = None
            else:
                previous = self.filtered.get(name, value)
                smooth[name] = previous + alpha * (value - previous)
                self.filtered[name] = smooth[name]
        eye = smooth["ear"] is not None and smooth["ear"] < cfg.ear_threshold
        mouth = smooth["mar"] is not None and smooth["mar"] > cfg.mar_threshold
        head = (
            smooth["pitch"] is not None and smooth["pitch"] - cfg.pitch_offset > cfg.pitch_threshold
        ) or (smooth["head_drop"] is not None and smooth["head_drop"] > cfg.head_drop_threshold)
        for name, active in (("eye", eye), ("mouth", mouth), ("head", head)):
            if not active:
                self.since[name] = None
            elif self.since[name] is None:
                self.since[name] = timestamp
            self.durations[name] = 0.0 if self.since[name] is None else timestamp - self.since[name]
        d = self.durations
        if max(d["eye"], d["head"]) >= cfg.sleep_seconds:
            state, reason = "sleeping", "Nhắm mắt hoặc gục đầu kéo dài"
        elif max(d["eye"], d["head"]) >= cfg.nod_seconds:
            state, reason = "nodding", "Nhắm mắt hoặc cúi đầu liên tục"
        elif d["mouth"] >= cfg.yawn_seconds:
            state, reason = "yawning", "Miệng mở rộng kéo dài"
        elif smooth["ear"] is not None:
            state, reason = "awake", "Chưa có dấu hiệu kéo dài vượt ngưỡng"
        else:
            state, reason = "unknown", "Không đủ landmark mắt; tiếp tục theo dõi tư thế"
        score = (
            0.4 * min(d["eye"] / cfg.sleep_seconds, 1)
            + 0.2 * min(d["mouth"] / cfg.yawn_seconds, 1)
            + 0.4 * min(d["head"] / cfg.sleep_seconds, 1)
        )
        return {**smooth, "state": state, "label": LABELS[state], "reason": reason,
                "score": round(score, 3), "eye_seconds": d["eye"],
                "yawn_seconds": d["mouth"], "head_seconds": d["head"],
                "face_visible": signals.face_visible, "pose_visible": signals.pose_visible}


def aspect_ratio(points: np.ndarray) -> float | None:
    """Six points ordered: left corner, upper pair, right corner, lower pair."""
    width = np.linalg.norm(points[0] - points[3])
    if width < 1e-6:
        return None
    return float((np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])) / (2 * width))


def head_angles(points: np.ndarray, width: int, height: int) -> tuple[float, float] | None:
    """Approximate pitch/yaw using full-frame pixels and a generic 3D face."""
    focal = float(max(width, height))
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, translation = cv2.solvePnP(
        FACE_MODEL, np.ascontiguousarray(points, dtype=np.float64), camera,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or translation[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(FACE_MODEL, rotation, translation, camera, np.zeros((4, 1)))
    error = np.linalg.norm(projected[:, 0] - points, axis=1).mean()
    if error > max(4.0, np.linalg.norm(points[2] - points[3]) * 0.15):
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
    yaw = math.degrees(math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0])))
    if not np.isfinite([pitch, yaw]).all() or abs(pitch) > 85:
        return None
    return pitch, yaw


def clip_box(box, width: int, height: int, padding: float = 0.0):
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, min(width, int(x1 - dx))), max(0, min(height, int(y1 - dy))),
            max(0, min(width, int(math.ceil(x2 + dx)))), max(0, min(height, int(math.ceil(y2 + dy)))))


class ClassroomEngine:
    def __init__(self, cfg: ClassroomSettings):
        import mediapipe as mp
        import torch
        from ultralytics import YOLO

        if not hasattr(mp, "solutions"):
            raise RuntimeError("Demo dùng MediaPipe 0.10.14. Cài requirements-classroom.txt trong môi trường Python 3.10/3.11.")
        try:
            import lap  # noqa: F401; prevent Ultralytics installing dependencies during inference
        except ImportError as exc:
            raise RuntimeError("Thiếu ByteTrack dependency: python -m pip install lapx==0.5.9") from exc
        if not cfg.person_weights or not Path(cfg.person_weights).is_file():
            raise FileNotFoundError(
                f"Không tìm thấy weights YOLOv8-Pose: {cfg.person_weights}. "
                "Dùng cùng yolov8l-pose.pt với project wall-climbing."
            )
        if cfg.face_weights and not Path(cfg.face_weights).is_file():
            raise FileNotFoundError(f"Không tìm thấy weights khuôn mặt: {cfg.face_weights}")
        self.cfg, self.states = cfg, {}
        self.device = (0 if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        self.person = YOLO(cfg.person_weights)
        if self.person.task != "pose" or self.person.names.get(0) != "person":
            raise ValueError("Model phải là YOLOv8-Pose COCO, class 0 = person.")
        self.face_model = YOLO(cfg.face_weights) if cfg.face_weights else None
        if self.face_model is not None:
            if self.face_model.task != "detect" or not any("face" in name.lower() for name in self.face_model.names.values()):
                raise ValueError("Weights khuôn mặt phải là YOLOv8-Face có class face.")
        self.resources = ExitStack()
        try:
            self.face_detector = self.resources.enter_context(mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5))
            self.mesh = self.resources.enter_context(mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5))
        except BaseException:
            self.resources.close()
            raise

    def close(self):
        self.resources.close()

    def _face_box(self, crop, nose):
        height, width = crop.shape[:2]
        candidates = []
        if self.face_model is not None:
            classes = [key for key, name in self.face_model.names.items() if "face" in name.lower()]
            result = self.face_model.predict(crop, classes=classes, conf=0.4, imgsz=320, device=self.device, verbose=False)[0]
            if result.boxes is not None:
                candidates = result.boxes.xyxy.cpu().numpy().tolist()
        else:
            result = self.face_detector.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for detection in result.detections or []:
                b = detection.location_data.relative_bounding_box
                candidates.append((b.xmin * width, b.ymin * height, (b.xmin + b.width) * width, (b.ymin + b.height) * height))
        valid = []
        for candidate in candidates:
            box = clip_box(candidate, width, height)
            x1, y1, x2, y2 = box
            if min(x2 - x1, y2 - y1) < self.cfg.min_face_pixels:
                continue
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            if nose is not None:
                distance = float(np.linalg.norm(center - nose) / max(x2 - x1, y2 - y1))
                if distance <= 1.0:
                    valid.append((distance, box))
            elif center[1] < height * 0.65:
                valid.append((0.0, box))
        # Without an anatomical anchor, multiple faces in an overlapping person
        # crop are ambiguous; do not assign another student's eyes to this ID.
        if not valid or (nose is None and len(valid) > 1):
            return None
        valid.sort(key=lambda item: item[0])
        return clip_box(valid[0][1], width, height, padding=0.25)

    def _signals(self, frame, box, canvas, pose_xy, pose_conf):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        height, width = crop.shape[:2]
        signals = Signals()

        # YOLOv8-Pose COCO indices used here:
        # 0=nose, 5=left shoulder, 6=right shoulder.
        # This preserves the original head-drop formula; only the pose provider changes.
        nose = None
        points = np.asarray(pose_xy, dtype=np.float32) if pose_xy is not None else np.empty((0, 2), np.float32)
        confidences = (
            np.asarray(pose_conf, dtype=np.float32)
            if pose_conf is not None
            else np.ones((len(points),), dtype=np.float32)
        )

        def visible(i):
            return (
                i < len(points)
                and i < len(confidences)
                and np.isfinite(points[i]).all()
                and float(confidences[i]) >= 0.6
                and 0 <= points[i, 0] < frame.shape[1]
                and 0 <= points[i, 1] < frame.shape[0]
            )

        if visible(0):
            nose = points[0] - np.array([x1, y1], dtype=np.float32)

        if all(visible(i) for i in (0, 5, 6)):
            shoulder_width = np.linalg.norm(points[5] - points[6])
            if shoulder_width >= 15:
                signals.pose_visible = True
                signals.head_drop = float(
                    (points[0, 1] - (points[5, 1] + points[6, 1]) / 2) / shoulder_width
                )

        if self.cfg.draw_landmarks and len(points) >= 13:
            # COCO skeleton subset around head, shoulders, arms and torso.
            for a, b in (
                (0, 5), (0, 6), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
            ):
                if visible(a) and visible(b):
                    pa, pb = points[[a, b]].astype(int)
                    cv2.line(canvas, tuple(pa), tuple(pb), (230, 180, 30), 2)

        face_box = self._face_box(crop, nose)
        if face_box is None:
            return signals
        fx1, fy1, fx2, fy2 = face_box
        face_crop = crop[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            return signals
        scale = max(1.0, 256 / max(face_crop.shape[:2]))
        enlarged = cv2.resize(face_crop, None, fx=scale, fy=scale) if scale > 1 else face_crop
        mesh = self.mesh.process(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
        if not mesh.multi_face_landmarks:
            return signals
        points = np.array([(p.x * (fx2 - fx1) + fx1 + x1, p.y * (fy2 - fy1) + fy1 + y1)
                           for p in mesh.multi_face_landmarks[0].landmark])
        signals.face_visible = True
        eyes = [aspect_ratio(points[list(indices)]) for indices in (LEFT_EYE, RIGHT_EYE)]
        signals.ear = sum(eyes) / 2 if all(value is not None for value in eyes) else None
        mouth_width = np.linalg.norm(points[78] - points[308])
        if mouth_width >= 2:
            signals.mar = float(np.linalg.norm(points[13] - points[14]) / mouth_width)
        angles = head_angles(points[list(PNP_IDS)], frame.shape[1], frame.shape[0])
        if angles is not None and abs(angles[1]) < 60:
            signals.pitch = angles[0]
        elif angles is not None:
            signals.ear = signals.mar = None
        if self.cfg.draw_landmarks:
            cv2.rectangle(canvas, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (255, 210, 80), 1)
            for i in (*LEFT_EYE, *RIGHT_EYE, 13, 14, 78, 308):
                cv2.circle(canvas, tuple(points[i].astype(int)), 2, (80, 255, 255), -1)
        return signals

    def process(self, frame: np.ndarray, timestamp: float):
        cfg = self.cfg
        canvas = frame.copy()
        tracked_people = track_people(
            self.person,
            frame,
            ROOT / "pose_bytetrack.yaml",
            conf=cfg.confidence,
            imgsz=cfg.image_size,
            max_det=cfg.max_people,
            device=self.device,
        )
        rows, seen = [], set()
        for tracked in tracked_people:
            raw_box = tracked.bbox
            track_id = tracked.track_id
            confidence = tracked.confidence
            box = clip_box(raw_box, frame.shape[1], frame.shape[0])
            if min(box[2] - box[0], box[3] - box[1]) < 16:
                continue
            seen.add(track_id)
            state = self.states.setdefault(track_id, StudentState())
            row = state.update(
                self._signals(
                    frame,
                    box,
                    canvas,
                    tracked.keypoints_xy,
                    tracked.keypoint_conf,
                ),
                timestamp,
                cfg,
            )
            row.update(id=track_id, timestamp=round(timestamp, 4), confidence=round(confidence, 3))
            rows.append(row)
            color = COLORS[row["state"]]
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f'ID {track_id} | {row["state"].upper()}'
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # Missing observations break continuous evidence immediately, even if
        # ByteTrack later recovers the same ID. Forget stale states to bound RAM.
        for track_id in list(self.states):
            if track_id not in seen:
                del self.states[track_id]
        return canvas, rows


def encode_browser_video(source: Path, destination: Path) -> bool:
    import imageio_ffmpeg

    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
                       check=True, capture_output=True, timeout=180)
        return destination.is_file() and destination.stat().st_size > 0
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return False


class InferenceRun:
    """One Streamlit session's resources; step() keeps Start/Stop responsive."""

    def __init__(self, source: str | int | bytes, cfg: ClassroomSettings, suffix=".mp4"):
        self.temp = tempfile.TemporaryDirectory(prefix="classroom_demo_")
        self.directory = Path(self.temp.name)
        self.capture = self.writer = self.engine = None
        self.closed = False
        self.cfg, self.camera = cfg, isinstance(source, int)
        self.frame_count = self.inference_count = 0
        self.preview, self.rows = None, []
        self.people = {}
        self.last_timestamp = -1.0
        self.started = time.perf_counter()
        self.inference_seconds = 0.0
        self.csv_file = (self.directory / "observations.csv").open("w", newline="", encoding="utf-8-sig")
        self.csv_writer = None
        try:
            if isinstance(source, bytes):
                path = self.directory / ("input" + suffix)
                path.write_bytes(source)
                source = str(path)
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise ValueError("Không mở được nguồn. Kiểm tra video hoặc chỉ số webcam trên máy chạy Streamlit.")
            if self.camera:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps_fallback = not math.isfinite(fps) or fps <= 0
            self.fps = 25.0 if self.fps_fallback else fps
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total_frames = max(0, int(count)) if math.isfinite(count) and not self.camera else 0
            self.engine = ClassroomEngine(cfg)
            self.started = time.perf_counter()
        except BaseException:
            self.close()
            raise

    def step(self) -> bool:
        ok, frame = self.capture.read()
        if not ok:
            if self.frame_count == 0:
                raise ValueError("Nguồn mở được nhưng không giải mã được khung hình nào.")
            return False
        height, width = frame.shape[:2]
        scale = min(1.0, self.cfg.max_width / width)
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        frame = cv2.resize(frame, size) if size != (width, height) else frame
        if self.camera:
            timestamp = time.perf_counter() - self.started
        else:
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
                timestamp = max(self.frame_count / self.fps, self.last_timestamp + 1 / self.fps)
        self.last_timestamp = timestamp
        if self.frame_count % self.cfg.stride == 0:
            started = time.perf_counter()
            self.preview, self.rows = self.engine.process(frame, timestamp)
            self.inference_seconds += time.perf_counter() - started
            self.inference_count += 1
            for row in self.rows:
                if self.csv_writer is None:
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(row))
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(row)
                person = self.people.setdefault(row["id"], {"id": row["id"], "observations": 0, "face_observations": 0, **dict.fromkeys(LABELS, 0)})
                person["observations"] += 1
                person["face_observations"] += int(row["face_visible"])
                person[row["state"]] += 1
        if not self.camera:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.directory / "annotated_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
                if not self.writer.isOpened():
                    raise RuntimeError("OpenCV không tạo được video MP4.")
            # Repeat the last analysed image for skipped frames, preserving the
            # source duration while making the sampling rate explicit in the UI.
            self.writer.write(self.preview)
        self.frame_count += 1
        return True

    def finish(self, stopped=False) -> dict:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.csv_file.flush()
        csv_bytes = (self.directory / "observations.csv").read_bytes()
        if not csv_bytes:
            csv_bytes = "timestamp,id,state,label\n".encode("utf-8-sig")
        result = {
            "frames": self.frame_count, "analyses": self.inference_count,
            "inference_fps": self.inference_count / max(self.inference_seconds, 1e-9),
            "elapsed": time.perf_counter() - self.started, "source_fps": self.fps,
            "duration": max(0.0, self.last_timestamp) if self.camera else self.frame_count / self.fps,
            "people": list(self.people.values()),
            "csv": csv_bytes, "video": None, "browser_video": False,
            "stopped": stopped, "settings": asdict(self.cfg), "fps_fallback": self.fps_fallback,
        }
        if not self.camera and self.frame_count:
            raw, encoded = self.directory / "annotated_raw.mp4", self.directory / "annotated.mp4"
            result["browser_video"] = encode_browser_video(raw, encoded)
            result["video"] = (encoded if result["browser_video"] else raw).read_bytes()
        self.close()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.capture is not None:
                self.capture.release()
            if self.writer is not None:
                self.writer.release()
            if self.engine is not None:
                self.engine.close()
        finally:
            self.csv_file.close()
            self.temp.cleanup()

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()
