"""OpenCV real-time video stream ingestion and frame preprocessing."""

from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np


class VideoStreamIngestor:
    """Real-time high-efficiency video stream ingestion with OpenCV."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480, fps: int = 15):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def read_video(self, video_path: str) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self.preprocess_frame(frame)
        finally:
            cap.release()

    def read_image(self, image_path: str) -> np.ndarray:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.preprocess_frame(frame)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def extract_frames(self, video_path: str, output_dir: str, every_n: int = 5) -> list[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx, frame in enumerate(self.read_video(video_path)):
            if idx % every_n == 0:
                path = out / f"frame_{idx:05d}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                paths.append(str(path))
        return paths

    def simulate_camera_stream(self, image_paths: list[str], n_cycles: int = 3) -> Generator[np.ndarray, None, None]:
        for _ in range(n_cycles):
            for path in image_paths:
                yield self.read_image(path)

    def compute_motion_vectors(self, video_path: str) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        vectors = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                vectors.append({"frame": idx, "mean_motion": float(magnitude.mean()), "max_motion": float(magnitude.max())})
            prev_gray = gray
            idx += 1
        cap.release()
        return vectors

    def create_inventory_snapshot(self, frame: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                objects.append({"bbox": [x, y, x + w, y + h], "area": float(area)})
        return {"object_count": len(objects), "objects": objects, "frame_shape": list(frame.shape)}
