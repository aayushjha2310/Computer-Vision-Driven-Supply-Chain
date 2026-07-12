"""Albumentations bounding-box and segmentation-aware data augmentation."""

from typing import Any

import cv2
import numpy as np

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except (ImportError, OSError):
    A = None
    ALBUMENTATIONS_AVAILABLE = False


class VisionAugmentor:
    """Advanced bbox and segmentation-aware geometric and color augmentations."""

    def __init__(self):
        if ALBUMENTATIONS_AVAILABLE:
            self.train_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                    A.Rotate(limit=15, p=0.4),
                    A.RandomScale(scale_limit=0.2, p=0.4),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
            self.seg_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                    A.GridDistortion(p=0.2),
                ],
            )
        else:
            self.train_transform = None
            self.seg_transform = None

    def augment_with_bboxes(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[str],
    ) -> dict[str, Any]:
        if ALBUMENTATIONS_AVAILABLE and self.train_transform is not None:
            labels_idx = list(range(len(class_labels)))
            transformed = self.train_transform(image=image, bboxes=bboxes, class_labels=labels_idx)
            return {
                "image": transformed["image"],
                "bboxes": transformed["bboxes"],
                "class_labels": [class_labels[i] for i in transformed["class_labels"]],
            }
        return self._fallback_bbox_augment(image, bboxes, class_labels)

    def _fallback_bbox_augment(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[str],
    ) -> dict[str, Any]:
        img = image.copy()
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            w = img.shape[1]
            bboxes = [[w - b[2], b[1], w - b[0], b[3]] for b in bboxes]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return {"image": img, "bboxes": bboxes, "class_labels": class_labels}

    def augment_with_mask(self, image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if ALBUMENTATIONS_AVAILABLE and self.seg_transform is not None:
            transformed = self.seg_transform(image=image, mask=mask)
            return {"image": transformed["image"], "mask": transformed["mask"]}
        img = image.copy()
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        return {"image": img, "mask": mask}

    def create_augmented_dataset(
        self,
        images: list[np.ndarray],
        bboxes_list: list[list[list[float]]],
        labels_list: list[list[str]],
        n_augmentations: int = 3,
    ) -> list[dict[str, Any]]:
        augmented = []
        for img, bboxes, labels in zip(images, bboxes_list, labels_list):
            for _ in range(n_augmentations):
                result = self.augment_with_bboxes(img, bboxes, labels)
                augmented.append(result)
        return augmented

    def apply_color_jitter_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if ALBUMENTATIONS_AVAILABLE:
            jitter = A.Compose(
                [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0)]
            )
            return [jitter(image=img)["image"] for img in images]
        result = []
        for img in images:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            result.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        return result
