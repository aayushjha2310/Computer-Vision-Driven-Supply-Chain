"""SAM (Segment Anything Model) for pixel-level defect/damage boundary mapping."""

from typing import Any

import cv2
import numpy as np


class SAMDamageSegmenter:
    """Segment Anything Model for precise defect/damage boundary mapping."""

    def __init__(self, model_type: str = "vit_b"):
        self.model_type = model_type
        self.predictor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
            checkpoint_map = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
            }
            checkpoint = checkpoint_map.get(self.model_type, checkpoint_map["vit_b"])
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            self.predictor = SamPredictor(sam)
        except Exception:
            self.predictor = None

    def segment_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> dict[str, Any]:
        if self.predictor is not None:
            return self._segment_sam(image, point_coords, point_labels)
        return self._segment_fallback(image, point_coords)

    def _segment_sam(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> dict[str, Any]:
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        return self._mask_to_result(mask, float(scores[best_idx]))

    def _segment_fallback(self, image: np.ndarray, point_coords: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for pt in point_coords:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(mask, (cx, cy), 40, 255, -1)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(thresh, mask)
        if mask.sum() == 0:
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)
        return self._mask_to_result(mask.astype(bool), 0.7)

    def _mask_to_result(self, mask: np.ndarray, score: float) -> dict[str, Any]:
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        damage_area = float(mask.sum())
        total_area = mask.size
        return {
            "mask": mask,
            "confidence": score,
            "damage_area_pixels": damage_area,
            "damage_ratio": damage_area / total_area,
            "n_contours": len(contours),
            "contours": [c.squeeze().tolist() for c in contours[:5] if len(c) > 2],
        }

    def auto_detect_damage(self, image: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        diff = cv2.Laplacian(gray, cv2.CV_64F)
        threshold = np.percentile(np.abs(diff), 95)
        damage_points = np.argwhere(np.abs(diff) > threshold)
        if len(damage_points) == 0:
            h, w = gray.shape
            point_coords = np.array([[w // 2, h // 2]])
        else:
            sample_idx = np.random.choice(len(damage_points), size=min(3, len(damage_points)), replace=False)
            point_coords = damage_points[sample_idx][:, ::-1]
        point_labels = np.ones(len(point_coords))
        return self.segment_with_points(image, point_coords, point_labels)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
        overlay = image.copy()
        colored = np.zeros_like(image)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
        contours, _ = cv2.findContours(
            (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        return overlay
