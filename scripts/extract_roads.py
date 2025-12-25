from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from road_extractors import get_extractor


def _load_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load config files") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must define a mapping")
    return data


def _read_image(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))


def _save_mask(path: str, mask: np.ndarray) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".npy":
        np.save(out_path, mask)
        return

    if mask.dtype != np.uint8:
        scaled = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        scaled = mask

    try:
        import cv2  # type: ignore

        cv2.imwrite(str(out_path), scaled)
    except Exception:
        from PIL import Image

        Image.fromarray(scaled).save(out_path)


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract road masks from an image")
    parser.add_argument("--config", default="configs/road_extraction.yaml")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--return-prob", action="store_true")
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patches-per-edge", type=int, default=None)
    parser.add_argument("--sample-margin", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--road-threshold", type=float, default=None)
    parser.add_argument("--sam-mask-channel", type=int, default=None)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--downscale-max-size", type=int, default=None)
    parser.add_argument("--multi-scale", type=str, default=None)
    parser.add_argument("--multi-scale-fuse", type=str, default=None)
    parser.add_argument("--min-component-area", type=int, default=None)
    parser.add_argument("--blur-kernel", type=int, default=None)
    parser.add_argument("--blur-sigma", type=float, default=None)
    parser.add_argument("--close-iters", type=int, default=None)
    parser.add_argument("--open-iters", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--skeletonize", action="store_true")
    parser.add_argument("--prune-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    backend_name = args.backend or cfg.get("backend", "sam_road")
    backend_cfg = cfg.get(backend_name, {}) if isinstance(cfg, dict) else {}

    overrides = {
        "device": args.device,
        "return_prob": args.return_prob or None,
        "patch_size": args.patch_size,
        "patches_per_edge": args.patches_per_edge,
        "sample_margin": args.sample_margin,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "road_threshold": args.road_threshold,
        "downscale_max_size": args.downscale_max_size,
        "multi_scale_max_sizes": (
            [int(x) for x in args.multi_scale.split(",") if x.strip()]
            if args.multi_scale
            else None
        ),
        "multi_scale_fuse": args.multi_scale_fuse,
        "mask_channel": args.sam_mask_channel,
        "auto_threshold": args.auto_threshold or None,
        "target_coverage": args.target_coverage,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "blur_kernel": args.blur_kernel,
        "blur_sigma": args.blur_sigma,
    }
    if any(
        value is not None
        for value in (
            args.min_component_area,
            args.close_iters,
            args.open_iters,
            args.kernel_size,
            args.prune_iters,
        )
    ):
        postprocess = dict(backend_cfg.get("postprocess", {}))
        if args.min_component_area is not None:
            postprocess["min_component_area"] = args.min_component_area
        if args.close_iters is not None:
            postprocess["close_iters"] = args.close_iters
        if args.open_iters is not None:
            postprocess["open_iters"] = args.open_iters
        if args.kernel_size is not None:
            postprocess["kernel_size"] = args.kernel_size
        if args.skeletonize:
            postprocess["skeletonize"] = True
        if args.prune_iters is not None:
            postprocess["prune_iters"] = args.prune_iters
        backend_cfg = dict(backend_cfg)
        backend_cfg["postprocess"] = postprocess

    extractor = get_extractor(backend_name, **_merge_overrides(backend_cfg, overrides))

    image = _read_image(args.image)
    mask = extractor.predict_mask(image)
    _save_mask(args.out, mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
