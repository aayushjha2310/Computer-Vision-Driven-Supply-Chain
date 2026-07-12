"""Gradio interactive web UI for field QA testing."""

import json
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from src.computer_vision.sam_segmentation import SAMDamageSegmenter
from src.computer_vision.vit_classifier import ViTProductClassifier
from src.computer_vision.yolo_tracker import YOLOAssetTracker
from src.utils.config import load_config


class SupplyChainGradioApp:
    """Interactive web-UI sandbox for warehouse vision QA testing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.yolo = YOLOAssetTracker(
            model_path=self.config["vision"]["yolo_model"],
            confidence=self.config["vision"]["confidence_threshold"],
        )
        self.vit = ViTProductClassifier(model_name=self.config["vision"]["vit_model"])
        self.sam = SAMDamageSegmenter()

    def analyze_image(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        if image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), "No image provided"
        detections = self.yolo.detect(image)
        annotated = self.yolo.annotate_frame(image, detections)
        labels = self.vit.extract_shipment_labels(image)
        damage = self.sam.auto_detect_damage(image)
        overlay = self.sam.overlay_mask(annotated, damage["mask"])
        report = {
            "detections": len(detections),
            "objects": [{"class": d["class_name"], "conf": round(d["confidence"], 3)} for d in detections],
            "shipment_labels": labels,
            "damage_ratio": round(damage["damage_ratio"], 4),
            "damage_detected": damage["damage_ratio"] > 0.01,
        }
        return overlay, json.dumps(report, indent=2)

    def launch(self, share: bool = False) -> None:
        with gr.Blocks(title="Supply Chain Vision QA") as demo:
            gr.Markdown("# Vision-Driven Supply Chain Intelligence")
            gr.Markdown("Upload warehouse images for asset detection, classification, and damage inspection.")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Warehouse Image", type="numpy")
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Annotated Result")
                    output_json = gr.Textbox(label="Analysis Report", lines=15)
            analyze_btn.click(fn=self.analyze_image, inputs=[input_image], outputs=[output_image, output_json])
            gr.Markdown("### Capabilities: YOLO Detection | ViT Classification | SAM Damage Segmentation")
        host = self.config["serving"]["host"]
        port = self.config["serving"]["port"]
        demo.launch(server_name=host, server_port=port, share=share)


def main():
    app = SupplyChainGradioApp()
    app.launch()


if __name__ == "__main__":
    main()
