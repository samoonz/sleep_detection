"""Behavior checks without loading or downloading neural-network weights."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from classroom_core import (
    FACE_MODEL, ClassroomSettings, InferenceRun, Signals, StudentState,
    aspect_ratio, clip_box, head_angles,
)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ClassroomSettings(smoothing_seconds=0)
        self.state = StudentState()

    def observe(self, signals, until, step=0.1):
        result = None
        for timestamp in np.arange(0, until + step / 2, step):
            result = self.state.update(signals, float(timestamp), self.cfg)
        return result

    def test_normal_blink_does_not_trigger_nodding(self):
        result = self.observe(Signals(ear=0.1, face_visible=True), 0.2)
        self.assertEqual(result["state"], "awake")
        result = self.state.update(Signals(ear=0.3, face_visible=True), 0.3, self.cfg)
        self.assertEqual(result["eye_seconds"], 0)

    def test_closed_eyes_alone_trigger_nodding_and_sleep(self):
        self.assertEqual(self.observe(Signals(ear=0.1), 1.0)["state"], "nodding")
        self.state = StudentState()
        self.assertEqual(self.observe(Signals(ear=0.1), 3.0)["state"], "sleeping")

    def test_pose_alone_can_detect_sleep_with_hidden_face(self):
        result = self.observe(Signals(head_drop=0.2, pose_visible=True), 3.0)
        self.assertEqual(result["state"], "sleeping")
        self.assertFalse(result["face_visible"])

    def test_missing_face_alone_never_means_sleeping(self):
        self.assertEqual(self.observe(Signals(), 10)["state"], "unknown")

    def test_yawn_is_independent_of_eye_closure(self):
        self.assertEqual(self.observe(Signals(ear=0.3, mar=0.9), 1.0)["state"], "yawning")

    def test_missing_observation_breaks_continuous_closure(self):
        self.observe(Signals(ear=0.1), 0.7)
        self.state.update(Signals(), 0.8, self.cfg)
        result = self.state.update(Signals(ear=0.1), 0.9, self.cfg)
        self.assertEqual(result["eye_seconds"], 0)
        self.assertEqual(result["state"], "awake")

    def test_long_gap_and_rewind_reset_evidence(self):
        self.observe(Signals(ear=0.1), 0.7)
        self.assertEqual(self.state.update(Signals(ear=0.1), 8, self.cfg)["eye_seconds"], 0)
        self.assertEqual(self.state.update(Signals(ear=0.1), 0, self.cfg)["eye_seconds"], 0)

    def test_two_students_have_independent_histories(self):
        second = StudentState()
        self.observe(Signals(ear=0.1), 3.0)
        result = second.update(Signals(ear=0.3), 3.0, self.cfg)
        self.assertEqual(result["state"], "awake")
        self.assertEqual(result["eye_seconds"], 0)

    def test_sampling_rate_does_not_change_duration_threshold(self):
        for step in (0.05, 0.2, 0.5):
            self.state = StudentState()
            self.assertEqual(self.observe(Signals(pitch=40), 3.0, step)["state"], "sleeping")

    def test_camera_pitch_offset(self):
        self.cfg = replace(self.cfg, pitch_offset=25)
        self.assertEqual(self.observe(Signals(ear=0.3, pitch=30), 3.0)["state"], "awake")

    def test_nan_landmarks_are_treated_as_missing(self):
        result = self.observe(Signals(ear=float("nan"), pitch=float("inf")), 3)
        self.assertEqual(result["state"], "unknown")

    def test_ema_does_not_reuse_missing_signal(self):
        cfg = ClassroomSettings()
        self.state.update(Signals(ear=0.1), 0, cfg)
        self.state.update(Signals(), 0.1, cfg)
        result = self.state.update(Signals(ear=0.3), 0.2, cfg)
        self.assertAlmostEqual(result["ear"], 0.3)


class GeometryTests(unittest.TestCase):
    def test_ear_scale_invariance_and_zero_width(self):
        eye = np.array([(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)], dtype=float)
        self.assertAlmostEqual(aspect_ratio(eye), 0.5)
        self.assertAlmostEqual(aspect_ratio(eye * 8), 0.5)
        self.assertIsNone(aspect_ratio(np.zeros((6, 2))))

    def test_signed_pitch_with_off_center_face(self):
        camera = np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=float)
        for pitch in (-35, 0, 35):
            points, _ = cv2.projectPoints(FACE_MODEL, np.array([np.deg2rad(pitch), 0, 0]),
                                         np.array([150.0, -40.0, 700.0]), camera, np.zeros((4, 1)))
            result = head_angles(points[:, 0], 1280, 720)
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result[0], pitch, places=3)

    def test_crops_are_clipped_to_frame(self):
        self.assertEqual(clip_box((-10, -20, 120, 200), 100, 80), (0, 0, 100, 80))


class VideoTests(unittest.TestCase):
    def test_stride_preserves_frame_count_and_cleans_up_resources(self):
        timestamps = []

        class FakeEngine:
            def __init__(self, cfg):
                self.closed = False

            def process(self, frame, timestamp):
                timestamps.append(timestamp)
                return frame, []

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(10):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            with patch("classroom_core.ClassroomEngine", FakeEngine):
                run = InferenceRun(str(source), ClassroomSettings(stride=3))
                engine, temporary = run.engine, run.directory
                try:
                    while run.step():
                        pass
                    result = run.finish()
                finally:
                    run.close()
            self.assertEqual(result["frames"], 10)
            self.assertEqual(result["analyses"], 4)
            self.assertAlmostEqual(result["duration"], 1.0)
            np.testing.assert_allclose(timestamps, [0, 0.3, 0.6, 0.9])
            self.assertTrue(engine.closed)
            self.assertFalse(temporary.exists())
            self.assertTrue(result["browser_video"])
            output = Path(directory) / "result.mp4"
            output.write_bytes(result["video"])
            capture = cv2.VideoCapture(str(output))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            capture.release()
            self.assertIn("timestamp,id,state,label", result["csv"].decode("utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
