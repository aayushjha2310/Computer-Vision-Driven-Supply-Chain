"""Diffusers synthetic edge-case imagery generation for anomaly training."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class SyntheticEdgeCaseGenerator:
    """Generatively fabricate rare, synthetic edge-case imagery for anomaly training."""

    EDGE_CASES = [
        "damaged_package", "spilled_contents", "mislabeled_crate",
        "overstacked_pallet", "hazmat_leak", "temperature_breach",
    ]

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
            )
            self.pipeline.set_progress_bar_config(disable=True)
        except (ImportError, OSError, Exception):
            self.pipeline = None

    def generate(self, prompt: str, n_images: int = 1, seed: int = 42) -> list[Image.Image]:
        if self.pipeline is not None:
            import torch
            generator = torch.Generator().manual_seed(seed)
            result = self.pipeline(prompt, num_images_per_prompt=n_images, generator=generator)
            return result.images
        return [self._generate_procedural(prompt, seed + i) for i in range(n_images)]

    def _generate_procedural(self, prompt: str, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        img = Image.new("RGB", (512, 512), color=(220, 220, 215))
        draw = ImageDraw.Draw(img)
        if "damaged" in prompt.lower():
            for _ in range(int(rng.integers(3, 8))):
                x, y = int(rng.integers(50, 450)), int(rng.integers(50, 450))
                draw.rectangle([x, y, x + 60, y + 40], fill=(180, 60, 40))
                draw.line([(x, y), (x + 60, y + 40)], fill=(100, 20, 10), width=3)
        elif "spill" in prompt.lower():
            draw.ellipse([150, 200, 350, 400], fill=(200, 180, 50))
            for _ in range(20):
                x, y = int(rng.integers(100, 400)), int(rng.integers(300, 480))
                draw.ellipse([x, y, x + 10, y + 10], fill=(180, 160, 40))
        elif "hazmat" in prompt.lower():
            draw.rectangle([180, 150, 330, 350], fill=(220, 220, 0), outline=(0, 0, 0), width=3)
            draw.polygon([(200, 380), (250, 420), (300, 380)], fill=(200, 50, 50))
        else:
            draw.rectangle([160, 180, 350, 380], fill=(100, 140, 180), outline=(0, 0, 0), width=2)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img

    def generate_edge_case_dataset(self, output_dir: str, n_per_case: int = 3) -> dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        generated = {}
        for case in self.EDGE_CASES:
            prompt = f"warehouse {case.replace('_', ' ')}, industrial photography, realistic"
            images = self.generate(prompt, n_images=n_per_case, seed=hash(case) % 10000)
            paths = []
            for i, img in enumerate(images):
                path = out / f"{case}_{i:03d}.png"
                img.save(path)
                paths.append(str(path))
            generated[case] = paths
        return {"cases": generated, "total_images": sum(len(v) for v in generated.values())}
