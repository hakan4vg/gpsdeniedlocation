from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import RoadExtractor
from .postprocess import postprocess_mask


class SamRoadExtractor(RoadExtractor):
    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        sam_checkpoint_path: str | None = None,
        device: str = "cpu",
        patch_size: int | None = None,
        patches_per_edge: int | None = None,
        sample_margin: int | None = None,
        stride: int | None = None,
        batch_size: int | None = None,
        road_threshold: float | None = None,
        downscale_max_size: int | None = None,
        multi_scale_max_sizes: list[int] | None = None,
        multi_scale_fuse: str = "max",
        return_prob: bool = False,
        mask_channel: int = 1,
        auto_threshold: bool = False,
        target_coverage: float = 0.02,
        threshold_min: float = 0.05,
        threshold_max: float = 0.9,
        blur_kernel: int | None = None,
        blur_sigma: float | None = None,
        postprocess: dict[str, Any] | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sam_root = repo_root / "third_party" / "sam_road"
        sam_pkg_root = sam_root / "sam"
        for path in (sam_root, sam_pkg_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from utils import load_config  # type: ignore
        from model import SAMRoad  # type: ignore

        if config_path is None:
            config_path = str(sam_root / "config" / "toponet_vitb_512_cityscale.yaml")
        if checkpoint_path is None:
            checkpoint_path = str(repo_root / "models" / "cityscale_vitb_512_e10.ckpt")
        if sam_checkpoint_path is None:
            sam_checkpoint_path = str(repo_root / "models" / "sam_vit_b_01ec64.pth")

        config = load_config(config_path)
        if patch_size is not None:
            config.PATCH_SIZE = int(patch_size)
        if patches_per_edge is not None:
            config.INFER_PATCHES_PER_EDGE = int(patches_per_edge)
        if sample_margin is not None:
            config.SAMPLE_MARGIN = int(sample_margin)
        if batch_size is not None:
            config.INFER_BATCH_SIZE = int(batch_size)
        if road_threshold is not None:
            config.ROAD_THRESHOLD = float(road_threshold)
        config.SAM_CKPT_PATH = str(sam_checkpoint_path)

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        self.config = config
        self.patch_size = int(config.PATCH_SIZE)
        self.patches_per_edge = int(config.INFER_PATCHES_PER_EDGE)
        self.sample_margin = int(config.SAMPLE_MARGIN)
        self.stride = stride
        self.batch_size = int(config.INFER_BATCH_SIZE)
        self.road_threshold = float(config.ROAD_THRESHOLD)
        self.downscale_max_size = downscale_max_size
        self.multi_scale_max_sizes = (
            [int(x) for x in multi_scale_max_sizes] if multi_scale_max_sizes else None
        )
        self.multi_scale_fuse = multi_scale_fuse
        self.return_prob = return_prob
        self.mask_channel = int(mask_channel)
        self.auto_threshold = bool(auto_threshold)
        self.target_coverage = float(target_coverage)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.postprocess_cfg = postprocess or {}

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"sam_road checkpoint not found: {checkpoint_file}")
        sam_ckpt_file = Path(sam_checkpoint_path)
        if not sam_ckpt_file.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt_file}")

        self.model = SAMRoad(config)
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected an HxWxC image with 3 channels")
        image = image[:, :, :3]
        image = image.astype(np.uint8, copy=False)

        if self.multi_scale_max_sizes:
            masks = [
                self._predict_prob_mask(image, max_size)
                for max_size in self.multi_scale_max_sizes
            ]
            if self.multi_scale_fuse == "mean":
                fused = np.mean(np.stack(masks, axis=0), axis=0)
            else:
                fused = np.max(np.stack(masks, axis=0), axis=0)
            if self.return_prob:
                return fused.astype(np.float32)
            threshold = self._select_threshold(fused)
            return postprocess_mask(fused, threshold=threshold, **self.postprocess_cfg)

        mask = self._predict_prob_mask(image, self.downscale_max_size)
        if self.return_prob:
            return mask.astype(np.float32)
        threshold = self._select_threshold(mask)
        return postprocess_mask(mask, threshold=threshold, **self.postprocess_cfg)

    def _predict_prob_mask(self, image: np.ndarray, max_size: int | None) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]

        resized, scale = self._maybe_downscale(image, max_size)
        resized = self._maybe_blur(resized)
        padded, padded_h, padded_w = self._pad_to_patch(resized)

        mask = self._infer_mask(padded)
        mask = mask[:padded_h, :padded_w]

        if scale != 1.0:
            mask = self._resize_mask(mask, orig_w, orig_h)
        else:
            mask = mask[:orig_h, :orig_w]
        return mask.astype(np.float32)

    def _infer_mask(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        xs, ys = self._patch_positions(width, height)
        patches = [(x, y) for y in ys for x in xs]

        fused = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        counter = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        for offset in range(0, len(patches), self.batch_size):
            batch = patches[offset : offset + self.batch_size]
            batch_imgs = []
            for x0, y0 in batch:
                patch = image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size, :]
                batch_imgs.append(patch)
            batch_tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(self.device)

            with torch.inference_mode():
                mask_scores, _ = self.model.infer_masks_and_img_features(batch_tensor)
            channel = self.mask_channel
            if channel not in (0, 1):
                raise ValueError(f"mask_channel must be 0 or 1, got {channel}")
            road_scores = mask_scores[:, :, :, channel]

            for i, (x0, y0) in enumerate(batch):
                x1 = x0 + self.patch_size
                y1 = y0 + self.patch_size
                fused[y0:y1, x0:x1] += road_scores[i]
                counter[y0:y1, x0:x1] += 1.0

        counter = torch.clamp(counter, min=1.0)
        fused = fused / counter
        return fused.cpu().numpy()

    def _select_threshold(self, mask: np.ndarray) -> float:
        if not self.auto_threshold:
            return float(self.road_threshold)
        if mask.size == 0:
            return float(self.road_threshold)
        coverage = max(0.0, min(1.0, self.target_coverage))
        if coverage <= 0.0:
            return float(self.road_threshold)
        flat = mask.reshape(-1)
        try:
            threshold = float(np.quantile(flat, 1.0 - coverage))
        except Exception:
            threshold = float(self.road_threshold)
        threshold = max(self.threshold_min, min(self.threshold_max, threshold))
        return float(threshold)

    def _patch_positions(self, width: int, height: int) -> tuple[list[int], list[int]]:
        if self.stride is not None and self.stride > 0:
            xs = self._stride_positions(width, self.stride)
            ys = self._stride_positions(height, self.stride)
            return xs, ys

        xs = self._linspace_positions(width, self.patches_per_edge)
        ys = self._linspace_positions(height, self.patches_per_edge)
        return xs, ys

    def _linspace_positions(self, length: int, count: int) -> list[int]:
        if length <= self.patch_size:
            return [0]
        sample_min = self.sample_margin
        sample_max = length - (self.patch_size + self.sample_margin)
        if sample_max < sample_min:
            sample_min = 0
            sample_max = length - self.patch_size
        if count <= 1 or sample_max == sample_min:
            return [int(round((sample_min + sample_max) / 2))]
        return [int(round(x)) for x in np.linspace(sample_min, sample_max, count)]

    def _stride_positions(self, length: int, stride: int) -> list[int]:
        if length <= self.patch_size:
            return [0]
        positions = list(range(0, length - self.patch_size + 1, stride))
        if positions[-1] != length - self.patch_size:
            positions.append(length - self.patch_size)
        return positions

    def _maybe_downscale(self, image: np.ndarray, max_size: int | None) -> tuple[np.ndarray, float]:
        if max_size is None:
            return image, 1.0
        height, width = image.shape[:2]
        max_dim = max(height, width)
        if max_dim <= max_size:
            return image, 1.0
        scale = max_size / max_dim
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return self._resize_image(image, new_w, new_h), scale

    def _pad_to_patch(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        pad_h = max(0, self.patch_size - height)
        pad_w = max(0, self.patch_size - width)
        if pad_h == 0 and pad_w == 0:
            return image, height, width
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return padded, height, width

    def _maybe_blur(self, image: np.ndarray) -> np.ndarray:
        if self.blur_kernel is None and self.blur_sigma is None:
            return image
        kernel = self.blur_kernel or 0
        sigma = self.blur_sigma or 0.0
        if kernel <= 0 and sigma <= 0:
            return image
        if kernel > 0 and kernel % 2 == 0:
            kernel += 1
        try:
            import cv2  # type: ignore

            ksize = (kernel, kernel) if kernel > 0 else (0, 0)
            return cv2.GaussianBlur(image, ksize, sigmaX=sigma, sigmaY=sigma)
        except Exception:
            from PIL import Image, ImageFilter

            pil_img = Image.fromarray(image)
            radius = sigma if sigma > 0 else max(1.0, kernel / 2.0)
            return np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)))

    def _resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        try:
            import cv2  # type: ignore

            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            return np.array(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR))

    def _resize_mask(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        try:
            import cv2  # type: ignore

            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        except Exception:
            from PIL import Image

            return np.array(Image.fromarray(mask).resize((width, height), resample=Image.NEAREST))
