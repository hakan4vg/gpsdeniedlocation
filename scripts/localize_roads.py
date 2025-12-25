from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Global verbose flag
_VERBOSE = False


def _log(msg: str) -> None:
    """Print diagnostic message if verbose mode is enabled."""
    if _VERBOSE:
        print(f"[DEBUG] {msg}", flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from road_extractors import get_extractor
from road_extractors.postprocess import postprocess_mask


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


def _transform_coords(
    src_crs: Any,
    dst_crs: Any,
    xs: list[float],
    ys: list[float],
) -> tuple[list[float], list[float]]:
    if src_crs is None or dst_crs is None or src_crs == dst_crs:
        return xs, ys
    try:
        import rasterio.warp
    except Exception:
        return xs, ys
    out_x, out_y = rasterio.warp.transform(src_crs, dst_crs, xs, ys)
    return list(out_x), list(out_y)


def _dataset_to_wgs84(crs: Any, xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    return _transform_coords(crs, "EPSG:4326", xs, ys)


def _wgs84_to_dataset(crs: Any, lons: list[float], lats: list[float]) -> tuple[list[float], list[float]]:
    return _transform_coords("EPSG:4326", crs, lons, lats)


def _pixel_to_wgs84(transform: Any, crs: Any, row: float, col: float) -> tuple[float, float]:
    import rasterio

    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    lons, lats = _dataset_to_wgs84(crs, [float(x)], [float(y)])
    return float(lons[0]), float(lats[0])


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))


def _downscale_image(image: np.ndarray, max_size: int | None) -> tuple[np.ndarray, float, float]:
    if max_size is None:
        return image, 1.0, 1.0
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= max_size:
        return image, 1.0, 1.0
    scale = max_size / max_dim
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    try:
        import cv2  # type: ignore

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        resized = np.array(Image.fromarray(image).resize((new_w, new_h)))
    return resized, width / new_w, height / new_h


def _downsample_mask(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == out_shape:
        return mask.astype(np.uint8)
    try:
        import cv2  # type: ignore

        resized = cv2.resize(
            mask.astype(np.uint8),
            (out_shape[1], out_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        from PIL import Image

        resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (out_shape[1], out_shape[0]), resample=Image.BILINEAR
            )
        )
    return (resized > 0).astype(np.uint8)


def _infer_satellite_path(drone_path: Path) -> Path:
    name = drone_path.stem
    if len(name) < 2 or not name[:2].isdigit():
        raise ValueError(f"Cannot infer satellite map from {drone_path.name}")
    flight_id = name[:2]
    sat_path = REPO_ROOT / "UAV_VisLoc_dataset" / flight_id / f"satellite{flight_id}.tif"
    if not sat_path.exists():
        raise FileNotFoundError(f"Satellite map not found: {sat_path}")
    return sat_path


def _load_satellite_tiff(
    path: Path,
    max_size: int | None,
    window: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, float, float, Any, int, int]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    with rasterio.open(path) as src:
        if window is None:
            origin_row = 0
            origin_col = 0
            height = src.height
            width = src.width
            transform = src.transform
            win = None
        else:
            row0, row1, col0, col1 = window
            origin_row = int(row0)
            origin_col = int(col0)
            height = max(1, int(row1 - row0))
            width = max(1, int(col1 - col0))
            win = Window(origin_col, origin_row, width, height)
            transform = window_transform(win, src.transform)
        if max_size is None:
            data = src.read(window=win, out_dtype=np.uint8)
            rgb = np.transpose(data[:3], (1, 2, 0))
            return rgb, 1.0, 1.0, transform, origin_row, origin_col
        max_dim = max(height, width)
        if max_dim <= max_size:
            data = src.read(window=win, out_dtype=np.uint8)
            rgb = np.transpose(data[:3], (1, 2, 0))
            return rgb, 1.0, 1.0, transform, origin_row, origin_col
        scale = max_size / max_dim
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        data = src.read(
            window=win,
            out_shape=(src.count, new_h, new_w),
            resampling=Resampling.bilinear,
            out_dtype=np.uint8,
        )
        rgb = np.transpose(data[:3], (1, 2, 0))
        scale_x = width / new_w
        scale_y = height / new_h
        return rgb, scale_x, scale_y, transform, origin_row, origin_col


def _scaled_transform(transform: Any, scale_x: float, scale_y: float) -> Any:
    from rasterio.transform import Affine

    return transform * Affine.scale(scale_x, scale_y)


def _load_bbox_for_satellite(sat_path: Path) -> tuple[float, float, float, float]:
    csv_path = REPO_ROOT / "UAV_VisLoc_dataset" / "satellite_ coordinates_range.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Satellite coordinate range file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mapname") == sat_path.name:
                north = float(row["LT_lat_map"])
                west = float(row["LT_lon_map"])
                south = float(row["RB_lat_map"])
                east = float(row["RB_lon_map"])
                return north, south, east, west
    raise ValueError(f"Satellite {sat_path.name} not found in coordinate range file")


def _build_osm_mask(
    sat_path: Path,
    out_shape: tuple[int, int],
    transform: Any,
    dst_crs: Any | None,
    cache_dir: Path,
    network_type: str,
    osm_dilate_px: int,
    bbox: tuple[float, float, float, float] | None = None,
    cache_key: str | None = None,
    cache_only: bool = False,
    osm_timeout: int | None = None,
    osm_rate_limit: bool | None = None,
) -> np.ndarray:
    import osmnx as ox
    import rasterio.features

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_key or sat_path.stem
    cache_name = (
        f"{cache_tag}_osm_v2_{out_shape[1]}x{out_shape[0]}_{network_type}_d{osm_dilate_px}.npy"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return np.load(cache_path)
    if cache_only:
        raise RuntimeError(
            "OSM cache missing and cache_only is set. "
            "Run once with network access or provide a cached mask."
        )

    if bbox is None:
        north, south, east, west = _load_bbox_for_satellite(sat_path)
    else:
        north, south, east, west = bbox
    if osm_timeout is not None:
        ox.settings.timeout = int(osm_timeout)
    else:
        ox.settings.timeout = 180
    if osm_rate_limit is not None:
        ox.settings.overpass_rate_limit = bool(osm_rate_limit)
    edges = None
    try:
        try:
            graph = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                network_type=network_type,
                simplify=True,
            )
        except TypeError:
            try:
                graph = ox.graph_from_bbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    network_type=network_type,
                    simplify=True,
                )
            except TypeError:
                graph = ox.graph_from_bbox(
                    (north, south, east, west),
                    network_type=network_type,
                    simplify=True,
                )
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    except Exception:
        tags = {"highway": True}
        try:
            edges = ox.features_from_bbox(
                bbox=(north, south, east, west),
                tags=tags,
            )
        except TypeError:
            try:
                edges = ox.features_from_bbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    tags=tags,
                )
            except TypeError:
                edges = ox.features_from_bbox((north, south, east, west), tags=tags)
    if edges is None or edges.empty:
        shapes = []
    else:
        edges = edges[edges.geometry.notnull()]
        if dst_crs is not None and getattr(edges, "crs", None) is not None and edges.crs != dst_crs:
            try:
                edges = edges.to_crs(dst_crs)
            except Exception:
                pass
        if not edges.empty:
            edges = edges[edges.geom_type.isin(["LineString", "MultiLineString"])]
        shapes = ((geom, 1) for geom in edges.geometry)

    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )

    if osm_dilate_px > 0:
        try:
            import cv2  # type: ignore

            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=osm_dilate_px)
        except Exception:
            pass

    np.save(cache_path, mask)
    return mask


def _parse_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    if value.strip().lower() in {"auto", "none"}:
        return None
    return [float(x) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return mask
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for rotation search") from exc
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for rotation search") from exc
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _prepare_match_mask(
    mask: np.ndarray,
    close_iters: int,
    open_iters: int,
    kernel_size: int,
    min_component_area: int,
    skeletonize: bool,
    prune_iters: int,
) -> np.ndarray:
    binary = _binary_mask(mask) * 255
    if close_iters <= 0 and open_iters <= 0 and min_component_area <= 0 and not skeletonize and prune_iters <= 0:
        return (binary > 0).astype(np.uint8)
    processed = postprocess_mask(
        binary,
        threshold=None,
        close_iters=close_iters,
        open_iters=open_iters,
        kernel_size=kernel_size,
        min_component_area=min_component_area,
        skeletonize=skeletonize,
        prune_iters=prune_iters,
    )
    return (processed > 0).astype(np.uint8)


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = _binary_mask(mask) * 255
    processed = postprocess_mask(binary, threshold=None, skeletonize=True)
    return (processed > 0).astype(np.uint8)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for distance transform") from exc
    return cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 3)


def _window_road_sum(
    map_mask: np.ndarray,
    template_shape: tuple[int, int],
    cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for template matching") from exc
    key = (int(template_shape[0]), int(template_shape[1]))
    if key in cache:
        return cache[key]
    if map_mask.shape[0] < key[0] or map_mask.shape[1] < key[1]:
        return None
    kernel = np.ones(key, dtype=np.float32)
    sums = cv2.matchTemplate(map_mask.astype(np.float32), kernel, cv2.TM_CCORR)
    cache[key] = sums
    return sums


def _top_k_from_score(score_map: np.ndarray, top_k: int) -> list[tuple[float, tuple[int, int]]]:
    flat = score_map.ravel()
    if flat.size == 0:
        return []
    valid = np.isfinite(flat)
    if not valid.any():
        return []
    valid_idx = np.nonzero(valid)[0]
    k = max(1, min(int(top_k), valid_idx.size))
    if k >= valid_idx.size:
        idxs = valid_idx[np.argsort(flat[valid_idx])]
    else:
        part = np.argpartition(flat[valid_idx], k - 1)[:k]
        idxs = valid_idx[part]
        idxs = idxs[np.argsort(flat[idxs])]
    width = score_map.shape[1]
    out: list[tuple[float, tuple[int, int]]] = []
    for idx in idxs:
        y = int(idx // width)
        x = int(idx % width)
        out.append((float(flat[idx]), (x, y)))
    return out


def _candidate_overlap_metrics(
    map_mask: np.ndarray,
    drone_mask: np.ndarray,
    offset: tuple[int, int],
    tolerance_px: int,
) -> dict[str, float | None]:
    x0, y0 = offset
    h, w = drone_mask.shape[:2]
    map_crop = map_mask[y0 : y0 + h, x0 : x0 + w]
    if map_crop.shape[0] != h or map_crop.shape[1] != w:
        return {"precision": None, "recall": None, "f1": None}
    drone_bin = _binary_mask(drone_mask)
    map_bin = _binary_mask(map_crop)
    if tolerance_px > 0:
        map_dil = _dilate_mask(map_bin, tolerance_px)
        drone_dil = _dilate_mask(drone_bin, tolerance_px)
    else:
        map_dil = map_bin
        drone_dil = drone_bin
    overlap_drone = float(np.logical_and(drone_bin, map_dil).sum())
    overlap_map = float(np.logical_and(map_bin, drone_dil).sum())
    drone_sum = float(drone_bin.sum())
    map_sum = float(map_bin.sum())
    precision = overlap_drone / drone_sum if drone_sum > 0 else None
    recall = overlap_map / map_sum if map_sum > 0 else None
    f1 = None
    if precision is not None and recall is not None:
        denom = precision + recall
        if denom > 0:
            f1 = 2.0 * precision * recall / denom
    return {"precision": precision, "recall": recall, "f1": f1}


def _candidate_distance_metrics(
    map_mask: np.ndarray,
    drone_mask: np.ndarray,
    offset: tuple[int, int],
) -> dict[str, float | None]:
    x0, y0 = offset
    h, w = drone_mask.shape[:2]
    map_crop = map_mask[y0 : y0 + h, x0 : x0 + w]
    if map_crop.shape[0] != h or map_crop.shape[1] != w:
        return {
            "forward_mean": None,
            "reverse_mean": None,
            "forward_p90": None,
            "reverse_p90": None,
            "chamfer_mean": None,
            "chamfer_p90": None,
        }
    drone_bin = _binary_mask(drone_mask)
    map_bin = _binary_mask(map_crop)
    if drone_bin.sum() == 0 or map_bin.sum() == 0:
        return {
            "forward_mean": None,
            "reverse_mean": None,
            "forward_p90": None,
            "reverse_p90": None,
            "chamfer_mean": None,
            "chamfer_p90": None,
        }
    dt_map = _distance_transform(map_bin)
    dt_drone = _distance_transform(drone_bin)
    forward_vals = dt_map[drone_bin > 0]
    reverse_vals = dt_drone[map_bin > 0]
    forward_mean = float(forward_vals.mean()) if forward_vals.size else None
    reverse_mean = float(reverse_vals.mean()) if reverse_vals.size else None
    forward_p90 = float(np.percentile(forward_vals, 90)) if forward_vals.size else None
    reverse_p90 = float(np.percentile(reverse_vals, 90)) if reverse_vals.size else None
    chamfer_mean = None
    if forward_mean is not None and reverse_mean is not None:
        chamfer_mean = 0.5 * (forward_mean + reverse_mean)
    chamfer_p90 = None
    if forward_p90 is not None and reverse_p90 is not None:
        chamfer_p90 = max(forward_p90, reverse_p90)
    return {
        "forward_mean": forward_mean,
        "reverse_mean": reverse_mean,
        "forward_p90": forward_p90,
        "reverse_p90": reverse_p90,
        "chamfer_mean": chamfer_mean,
        "chamfer_p90": chamfer_p90,
    }


def _extract_keypoints(mask: np.ndarray) -> dict[str, np.ndarray]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for keypoint extraction") from exc
    binary = _binary_mask(mask).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    neighbors = cv2.filter2D(binary, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    neighbors = neighbors - binary
    junctions = np.column_stack(np.where((binary > 0.5) & (neighbors >= 3.0)))
    endpoints = np.column_stack(np.where((binary > 0.5) & (neighbors == 1.0)))
    return {"junctions": junctions, "endpoints": endpoints}


def _match_keypoints(
    drone_points: np.ndarray,
    map_points: np.ndarray,
    tolerance_px: float,
) -> tuple[float | None, float | None]:
    if drone_points.size == 0 or map_points.size == 0:
        return None, None
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return None, None
    tree = cKDTree(map_points)
    dists, _ = tree.query(drone_points, distance_upper_bound=tolerance_px)
    valid = np.isfinite(dists)
    if not valid.any():
        return 0.0, None
    inlier_ratio = float(valid.mean())
    median_dist = float(np.median(dists[valid]))
    return inlier_ratio, median_dist


def _match_candidates(
    dist_map: np.ndarray,
    map_mask: np.ndarray,
    drone_mask: np.ndarray,
    scales: list[float],
    rotations: list[float],
    top_k: int,
    min_template_road_px: int,
    min_window_road_px: int,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for template matching") from exc
    matches: list[dict[str, Any]] = []
    map_mask_f = map_mask.astype(np.float32)
    sum_cache: dict[tuple[int, int], np.ndarray] = {}
    
    _log(f"_match_candidates: dist_map={dist_map.shape}, map_mask={map_mask.shape}, drone_mask={drone_mask.shape}")
    _log(f"_match_candidates: scales={scales}, rotations={rotations}")
    _log(f"_match_candidates: min_template_road_px={min_template_road_px}, min_window_road_px={min_window_road_px}")
    _log(f"_match_candidates: map_mask road pixels = {int(np.count_nonzero(map_mask > 0))}")
    _log(f"_match_candidates: drone_mask road pixels = {int(np.count_nonzero(drone_mask > 0))}")
    
    skipped_scale_zero = 0
    skipped_resize_none = 0
    skipped_base_road_px = 0
    skipped_map_sum_none = 0
    skipped_rot_too_big = 0
    skipped_rot_road_px = 0
    skipped_all_inf = 0
    total_candidates_found = 0
    
    for scale in scales:
        if scale <= 0:
            skipped_scale_zero += 1
            continue
        scaled_mask = _resize_mask(drone_mask, scale)
        if scaled_mask is None:
            skipped_resize_none += 1
            _log(f"  scale={scale:.3f}: _resize_mask returned None")
            continue
        scaled_mask = _binary_mask(scaled_mask)
        base_road_px = int(np.count_nonzero(scaled_mask > 0))
        if min_template_road_px and base_road_px < min_template_road_px:
            skipped_base_road_px += 1
            _log(f"  scale={scale:.3f}: base_road_px={base_road_px} < min={min_template_road_px}, skipping")
            continue
        map_sum = _window_road_sum(map_mask_f, scaled_mask.shape[:2], sum_cache)
        if map_sum is None:
            skipped_map_sum_none += 1
            _log(f"  scale={scale:.3f}: map_sum is None (template bigger than map?)")
            continue
        for angle in rotations:
            rot_mask = _rotate_mask(scaled_mask, angle)
            if rot_mask.shape[0] >= dist_map.shape[0] or rot_mask.shape[1] >= dist_map.shape[1]:
                skipped_rot_too_big += 1
                continue
            rot_mask = _binary_mask(rot_mask)
            road_px = int(np.count_nonzero(rot_mask > 0))
            if min_template_road_px and road_px < min_template_road_px:
                skipped_rot_road_px += 1
                continue
            dt_drone = _distance_transform(rot_mask).astype(np.float32)
            tmpl = rot_mask.astype(np.float32)
            forward_sum = cv2.matchTemplate(dist_map, tmpl, cv2.TM_CCORR)
            forward_mean = forward_sum / max(1.0, float(road_px))
            reverse_sum = cv2.matchTemplate(map_mask_f, dt_drone, cv2.TM_CCORR)
            reverse_mean = reverse_sum / np.maximum(map_sum, 1.0)
            chamfer = 0.5 * (forward_mean + reverse_mean)
            if min_window_road_px > 0:
                chamfer = chamfer.copy()
                chamfer[map_sum < float(min_window_road_px)] = np.inf
            candidates = _top_k_from_score(chamfer, top_k)
            if not candidates:
                skipped_all_inf += 1
            total_candidates_found += len(candidates)
            for score, offset in candidates:
                ox, oy = offset
                matches.append(
                    {
                        "score": float(score),
                        "scale": scale,
                        "rotation": angle,
                        "offset": offset,
                        "mask": rot_mask,
                        "road_px": road_px,
                        "map_road_px": float(map_sum[oy, ox]) if map_sum.size else None,
                        "forward_mean": float(forward_mean[oy, ox]),
                        "reverse_mean": float(reverse_mean[oy, ox]),
                    }
                )
    
    _log(f"_match_candidates SUMMARY:")
    _log(f"  skipped_scale_zero={skipped_scale_zero}")
    _log(f"  skipped_resize_none={skipped_resize_none}")
    _log(f"  skipped_base_road_px={skipped_base_road_px}")
    _log(f"  skipped_map_sum_none={skipped_map_sum_none}")
    _log(f"  skipped_rot_too_big={skipped_rot_too_big}")
    _log(f"  skipped_rot_road_px={skipped_rot_road_px}")
    _log(f"  skipped_all_inf (no valid positions)={skipped_all_inf}")
    _log(f"  total_candidates_found={total_candidates_found}")
    _log(f"  returning {min(len(matches), max(1, int(top_k)))} matches")
    
    matches.sort(key=lambda item: item["score"])
    return matches[: max(1, int(top_k))]


def _score_candidates(
    candidates: list[dict[str, Any]],
    map_mask: np.ndarray,
    tolerance_px: int,
    f1_weight: float,
) -> list[dict[str, Any]]:
    for candidate in candidates:
        metrics = _candidate_overlap_metrics(
            map_mask, candidate["mask"], candidate["offset"], tolerance_px
        )
        candidate.update(metrics)
        f1 = metrics.get("f1")
        base = float(candidate.get("score", 0.0))
        if f1 is None:
            candidate["score_combined"] = base
        else:
            candidate["score_combined"] = base + (1.0 - float(f1)) * float(f1_weight)
    candidates.sort(key=lambda item: float(item.get("score_combined", item.get("score", 0.0))))
    return candidates


def _window_positions(total: int, window: int, stride: int) -> list[int]:
    if window >= total:
        return [0]
    stride = max(1, stride)
    positions = list(range(0, total - window + 1, stride))
    end = total - window
    if not positions or positions[-1] != end:
        positions.append(end)
    return positions


def _windowed_match_candidates(
    dist_map: np.ndarray,
    map_mask: np.ndarray,
    drone_mask: np.ndarray,
    scales: list[float],
    rotations: list[float],
    top_k: int,
    min_template_road_px: int,
    min_window_road_px: int,
    window_pad_ratio: float,
    window_stride_ratio: float,
    base_scale: float | None,
) -> list[dict[str, Any]]:
    _log(f"_windowed_match_candidates: base_scale={base_scale}, scales={scales}")
    if base_scale is None or base_scale <= 0:
        base_scale = float(np.median(scales)) if scales else 0.0
        _log(f"  base_scale was None/<=0, now using median: {base_scale}")
    if base_scale <= 0:
        _log("  base_scale still <= 0, returning empty!")
        return []
    tmpl_h = max(2, int(round(drone_mask.shape[0] * base_scale)))
    tmpl_w = max(2, int(round(drone_mask.shape[1] * base_scale)))
    pad_h = max(0, int(round(tmpl_h * max(0.0, window_pad_ratio))))
    pad_w = max(0, int(round(tmpl_w * max(0.0, window_pad_ratio))))
    win_h = min(dist_map.shape[0], tmpl_h + 2 * pad_h)
    win_w = min(dist_map.shape[1], tmpl_w + 2 * pad_w)
    stride_y = max(1, int(round(tmpl_h * max(0.1, window_stride_ratio))))
    stride_x = max(1, int(round(tmpl_w * max(0.1, window_stride_ratio))))

    y_positions = _window_positions(dist_map.shape[0], win_h, stride_y)
    x_positions = _window_positions(dist_map.shape[1], win_w, stride_x)
    total_windows = len(y_positions) * len(x_positions)
    _log(f"  tmpl_size=({tmpl_h}, {tmpl_w}), win_size=({win_h}, {win_w}), stride=({stride_y}, {stride_x})")
    _log(f"  dist_map.shape={dist_map.shape}, total_windows={total_windows}")

    matches: list[dict[str, Any]] = []
    windows_with_matches = 0
    for y0 in y_positions:
        for x0 in x_positions:
            dist_win = dist_map[y0 : y0 + win_h, x0 : x0 + win_w]
            map_win = map_mask[y0 : y0 + win_h, x0 : x0 + win_w]
            win_matches = _match_candidates(
                dist_win,
                map_win,
                drone_mask,
                scales,
                rotations,
                top_k,
                min_template_road_px,
                min_window_road_px,
            )
            if win_matches:
                windows_with_matches += 1
            for match in win_matches:
                off_x, off_y = match["offset"]
                match["offset"] = (int(off_x + x0), int(off_y + y0))
                match["window_origin"] = (int(x0), int(y0))
                matches.append(match)
    
    _log(f"  windows_with_matches={windows_with_matches}/{total_windows}, total matches={len(matches)}")
    matches.sort(key=lambda item: item["score"])
    return matches[: max(1, int(top_k))]


def _scale_list_around(base: float, frac_range: float, steps: int) -> list[float]:
    if steps <= 1:
        return [base]
    lo = max(0.01, base * (1.0 - frac_range))
    hi = max(lo + 1e-6, base * (1.0 + frac_range))
    return list(np.linspace(lo, hi, steps))


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    try:
        import cv2  # type: ignore

        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(mask, kernel, iterations=iterations)
    except Exception:
        return mask


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.4,
    dilate_iters: int = 0,
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    mask = _dilate_mask(mask, dilate_iters)
    mask_bool = mask > 0
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(path)


def _rotate_image_expand(
    image: np.ndarray, angle: float, is_mask: bool = False, add_alpha: bool = False
) -> np.ndarray:
    """Rotate image with expansion. If add_alpha=True, returns RGBA with transparent background."""
    if angle == 0 and not add_alpha:
        return image
    if angle == 0 and add_alpha and image.ndim == 3 and image.shape[2] == 3:
        # Just add full-opacity alpha channel
        h, w = image.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        return np.dstack([image, alpha])

    try:
        import cv2  # type: ignore

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(mat[0, 0])
        sin = abs(mat[0, 1])
        new_w = int(round(h * sin + w * cos))
        new_h = int(round(h * cos + w * sin))
        mat[0, 2] += (new_w / 2.0) - center[0]
        mat[1, 2] += (new_h / 2.0) - center[1]
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        if add_alpha and image.ndim == 3 and image.shape[2] == 3:
            # Create alpha mask: 255 for original pixels, 0 for background
            alpha_orig = np.full((h, w), 255, dtype=np.uint8)
            alpha_rot = cv2.warpAffine(
                alpha_orig, mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0
            )
            # Rotate RGB with black border (will be masked by alpha)
            rgb_rot = cv2.warpAffine(
                image, mat, (new_w, new_h), flags=interp, borderValue=(0, 0, 0)
            )
            return np.dstack([rgb_rot, alpha_rot])

        if image.ndim == 2:
            border = 0
        else:
            border = 0 if is_mask else (0, 0, 0)
        return cv2.warpAffine(image, mat, (new_w, new_h), flags=interp, borderValue=border)
    except Exception:
        from PIL import Image

        if add_alpha and image.ndim == 3 and image.shape[2] == 3:
            pil = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
            resample = Image.BILINEAR
            rotated = pil.rotate(angle, resample=resample, expand=True, fillcolor=(0, 0, 0, 0))
            return np.array(rotated)

        if image.ndim == 2:
            pil = Image.fromarray(image.astype(np.uint8))
            fill = 0
        else:
            pil = Image.fromarray(image.astype(np.uint8))
            fill = 0 if is_mask else (0, 0, 0)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        rotated = pil.rotate(angle, resample=resample, expand=True, fillcolor=fill)
        return np.array(rotated)


def _write_geotiff(
    path: Path,
    image: np.ndarray,
    transform: Any,
    crs: Any,
    nodata: int | float | None = None,
) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        height, width = image.shape
        count = 1
        data = image
    else:
        height, width = image.shape[:2]
        count = image.shape[2]
        data = np.moveaxis(image, 2, 0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(data)


def _write_geojson_point(
    path: Path,
    lon: float,
    lat: float,
    props: dict[str, Any] | None = None,
) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props or {},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_qgis_project(path: Path, layers: list[dict[str, str]]) -> None:
    import xml.etree.ElementTree as ET

    qgis = ET.Element("qgis", version="3.28.0", projectname=path.stem)
    projectlayers = ET.SubElement(qgis, "projectlayers")
    for layer in layers:
        maplayer = ET.SubElement(projectlayers, "maplayer", type=layer["type"], name=layer["name"])
        ET.SubElement(maplayer, "id").text = layer["id"]
        ET.SubElement(maplayer, "datasource").text = layer["path"]
        ET.SubElement(maplayer, "layername").text = layer["name"]
        ET.SubElement(maplayer, "provider").text = layer["provider"]
    layer_tree = ET.SubElement(qgis, "layer-tree-group", name="")
    for layer in layers:
        ET.SubElement(layer_tree, "layer-tree-layer", id=layer["id"], name=layer["name"])
    xml_bytes = ET.tostring(qgis, encoding="utf-8", xml_declaration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xml_bytes)


def _export_qgis_outputs(
    out_dir: Path,
    drone_rgb: np.ndarray,
    drone_mask: np.ndarray | None,
    sat_path: Path,
    full_transform: Any,
    full_crs: Any,
    full_w: int,
    full_h: int,
    center_row: float,
    center_col: float,
    scale_full_x: float | None,
    scale_full_y: float | None,
    rotation_deg: float,
    lat: float | None,
    lon: float | None,
    use_osm: bool,
    osm_cache_dir: Path,
    osm_network_type: str,
    osm_dilate_px: int,
    osm_cache_only: bool,
    osm_timeout: int | None,
    osm_rate_limit: bool | None,
    full_osm_mask: np.ndarray | None,
    pad_px: int = 0,
) -> None:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform

    qgis_dir = out_dir / "qgis"
    qgis_dir.mkdir(parents=True, exist_ok=True)

    layers: list[dict[str, str]] = []
    if lat is not None and lon is not None:
        point_path = qgis_dir / "found_point.geojson"
        _write_geojson_point(point_path, lon, lat, {"source": "localize_roads"})
        layers.append(
            {
                "id": f"found_point_{uuid.uuid4().hex}",
                "name": "found_point",
                "path": str(point_path.resolve()),
                "type": "vector",
                "provider": "ogr",
            }
        )

    if scale_full_x is None or scale_full_y is None or scale_full_x <= 0 or scale_full_y <= 0:
        _log("Skipping QGIS raster export: missing/invalid scale.")
        if layers:
            _write_qgis_project(qgis_dir / "qgis_project.qgs", layers)
        return

    drone_rot = _rotate_image_expand(drone_rgb, rotation_deg, is_mask=False, add_alpha=True)
    mask_rot = None
    if drone_mask is not None:
        mask_bin = (drone_mask > 0).astype(np.uint8) * 255
        mask_rot = _rotate_image_expand(mask_bin, rotation_deg, is_mask=True)

    px_size_x = full_transform.a * float(scale_full_x)
    px_size_y = full_transform.e * float(scale_full_y)
    x_c, y_c = rasterio.transform.xy(full_transform, center_row, center_col, offset="center")
    x0 = x_c - (drone_rot.shape[1] * px_size_x) / 2.0
    y0 = y_c - (drone_rot.shape[0] * px_size_y) / 2.0
    drone_transform = Affine(px_size_x, 0.0, x0, 0.0, px_size_y, y0)

    drone_path = qgis_dir / "drone_raw.tif"
    # Write as RGBA GeoTIFF; alpha channel makes background transparent
    _write_geotiff(drone_path, drone_rot, drone_transform, full_crs)
    layers.append(
        {
            "id": f"drone_raw_{uuid.uuid4().hex}",
            "name": "drone_raw",
            "path": str(drone_path.resolve()),
            "type": "raster",
            "provider": "gdal",
        }
    )

    if mask_rot is not None:
        seg_path = qgis_dir / "segmented_roads.tif"
        _write_geotiff(seg_path, mask_rot.astype(np.uint8), drone_transform, full_crs, nodata=0)
        layers.append(
            {
                "id": f"segmented_roads_{uuid.uuid4().hex}",
                "name": "segmented_roads",
                "path": str(seg_path.resolve()),
                "type": "raster",
                "provider": "gdal",
            }
        )

    width_map_px = drone_rot.shape[1] * float(scale_full_x)
    height_map_px = drone_rot.shape[0] * float(scale_full_y)
    col0 = int(math.floor(center_col - width_map_px / 2.0)) - int(pad_px)
    row0 = int(math.floor(center_row - height_map_px / 2.0)) - int(pad_px)
    col1 = int(math.ceil(center_col + width_map_px / 2.0)) + int(pad_px)
    row1 = int(math.ceil(center_row + height_map_px / 2.0)) + int(pad_px)
    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(full_w, col1)
    row1 = min(full_h, row1)
    if col1 <= col0 or row1 <= row0:
        _log("Skipping QGIS satellite/OSM export: invalid window.")
        _write_qgis_project(qgis_dir / "qgis_project.qgs", layers)
        return
    width = int(col1 - col0)
    height = int(row1 - row0)

    window = Window(col0, row0, width, height)
    window_tr = window_transform(window, full_transform)
    sat_crop = _read_satellite_crop(sat_path, col0, row0, width, height, max_size=None)
    sat_path_out = qgis_dir / "satellite_crop.tif"
    _write_geotiff(sat_path_out, sat_crop, window_tr, full_crs)
    layers.append(
        {
            "id": f"satellite_crop_{uuid.uuid4().hex}",
            "name": "satellite_crop",
            "path": str(sat_path_out.resolve()),
            "type": "raster",
            "provider": "gdal",
        }
    )

    if use_osm:
        osm_crop = None
        if full_osm_mask is not None:
            osm_crop = np.array(full_osm_mask[row0:row1, col0:col1], dtype=np.uint8)
        else:
            left, bottom, right, top = window_bounds(window, full_transform)
            bbox = _bounds_to_wgs84(full_crs, left, bottom, right, top)
            cache_key = f"{sat_path.stem}_qgis_{row0}_{col0}_{width}x{height}"
            try:
                osm_crop = _build_osm_mask(
                    sat_path,
                    out_shape=(height, width),
                    transform=window_tr,
                    dst_crs=full_crs,
                    cache_dir=osm_cache_dir,
                    network_type=osm_network_type,
                    osm_dilate_px=osm_dilate_px,
                    bbox=bbox,
                    cache_key=cache_key,
                    cache_only=osm_cache_only,
                    osm_timeout=osm_timeout,
                    osm_rate_limit=osm_rate_limit,
                )
            except Exception as exc:
                _log(f"Skipping OSM export: {exc}")
                osm_crop = None
        if osm_crop is not None:
            osm_path = qgis_dir / "osm_roads.tif"
            osm_vis = (osm_crop > 0).astype(np.uint8) * 255
            _write_geotiff(osm_path, osm_vis, window_tr, full_crs, nodata=0)
            layers.append(
                {
                    "id": f"osm_roads_{uuid.uuid4().hex}",
                    "name": "osm_roads",
                    "path": str(osm_path.resolve()),
                    "type": "raster",
                    "provider": "gdal",
                }
            )

    if layers:
        order = [
            "found_point",
            "drone_raw",
            "satellite_crop",
            "osm_roads",
            "segmented_roads",
        ]
        layers.sort(key=lambda item: order.index(item["name"]) if item["name"] in order else len(order))
        _write_qgis_project(qgis_dir / "qgis_project.qgs", layers)


def _write_candidates_csv(path: Path, matches: list[dict[str, Any]]) -> None:
    if not matches:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "score",
                "score_combined",
                "scale",
                "rotation",
                "offset_x",
                "offset_y",
                "road_px",
                "map_road_px",
                "forward_mean",
                "reverse_mean",
                "precision",
                "recall",
                "f1",
            ]
        )
        for item in matches:
            off_x, off_y = item.get("offset", (None, None))
            writer.writerow(
                [
                    item.get("score"),
                    item.get("score_combined"),
                    item.get("scale"),
                    item.get("rotation"),
                    off_x,
                    off_y,
                    item.get("road_px"),
                    item.get("map_road_px"),
                    item.get("forward_mean"),
                    item.get("reverse_mean"),
                    item.get("precision"),
                    item.get("recall"),
                    item.get("f1"),
                ]
            )


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / h
    new_w = max(1, int(round(w * scale)))
    try:
        import cv2  # type: ignore

        return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(image).resize((new_w, height)))


def _resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    try:
        import cv2  # type: ignore

        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(image).resize((target_w, target_h)))


def _resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    try:
        import cv2  # type: ignore

        return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(mask).resize((target_w, target_h), resample=Image.NEAREST))


def _blend_images(image_a: np.ndarray, image_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image_a.shape != image_b.shape:
        raise ValueError("Blend images must have the same shape")
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    blended = alpha * a + (1.0 - alpha) * b
    return np.clip(blended, 0, 255).astype(np.uint8)


def _draw_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    if image.size == 0:
        return image
    try:
        import cv2  # type: ignore

        out = image.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness=thickness)
        return out
    except Exception:
        out = image.copy()
        h, w = out.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        out[y0 : y0 + thickness, x0 : x1 + 1] = color
        out[y1 - thickness + 1 : y1 + 1, x0 : x1 + 1] = color
        out[y0 : y1 + 1, x0 : x0 + thickness] = color
        out[y0 : y1 + 1, x1 - thickness + 1 : x1 + 1] = color
        return out


def _load_drone_metadata(drone_path: Path) -> dict[str, float] | None:
    flight_id = drone_path.stem[:2]
    csv_path = REPO_ROOT / "UAV_VisLoc_dataset" / flight_id / f"{flight_id}.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename") == drone_path.name:
                out: dict[str, float] = {}
                for key in ("lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"):
                    value = row.get(key)
                    if value is None or value == "":
                        continue
                    try:
                        out[key.lower()] = float(value)
                    except Exception:
                        continue
                return out
    return None


def _estimate_map_gsd_mpp(transform: Any, crs: Any, ref_row: float, ref_col: float) -> tuple[float, float, float]:
    if crs is None:
        mpp_x = abs(transform.a)
        mpp_y = abs(transform.e)
        avg = (mpp_x + mpp_y) / 2.0
        return avg, mpp_x, mpp_y
    try:
        lon0, lat0 = _pixel_to_wgs84(transform, crs, ref_row, ref_col)
        lon1, lat1 = _pixel_to_wgs84(transform, crs, ref_row, ref_col + 1.0)
        lon2, lat2 = _pixel_to_wgs84(transform, crs, ref_row + 1.0, ref_col)
        mpp_x = _haversine_km(lat0, lon0, lat1, lon1) * 1000.0
        mpp_y = _haversine_km(lat0, lon0, lat2, lon2) * 1000.0
        if mpp_x <= 0 or mpp_y <= 0:
            raise ValueError("Invalid GSD estimate")
        return (mpp_x + mpp_y) / 2.0, mpp_x, mpp_y
    except Exception:
        mpp_x = abs(transform.a)
        mpp_y = abs(transform.e)
        avg = (mpp_x + mpp_y) / 2.0
        return avg, mpp_x, mpp_y


def _estimate_drone_gsd_mpp(height_m: float, image_width: int, image_height: int, fov_deg: float) -> float | None:
    if height_m <= 0 or fov_deg <= 0 or image_width <= 0 or image_height <= 0:
        return None
    fov_rad = math.radians(fov_deg)
    ground_width = 2.0 * height_m * math.tan(fov_rad / 2.0)
    return ground_width / float(image_width)


def _estimate_dominant_angle(mask: np.ndarray, max_size: int | None = None) -> float | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if max_size is not None:
        h, w = mask.shape[:2]
        max_dim = max(h, w)
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=40, maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        angles.append(angle)
    if len(angles) < 5:
        return None
    return float(np.median(angles))


def _auto_rotation_list(
    drone_mask: np.ndarray,
    map_mask: np.ndarray,
    range_deg: float,
    step_deg: float,
    max_size: int | None,
) -> tuple[list[float], float | None, float | None, float | None]:
    if step_deg <= 0 or range_deg < 0:
        return [0.0], None, None, None
    drone_angle = _estimate_dominant_angle(drone_mask, max_size=max_size)
    map_angle = _estimate_dominant_angle(map_mask, max_size=max_size)
    if drone_angle is None or map_angle is None:
        if range_deg <= 0 or step_deg <= 0:
            return [0.0], drone_angle, map_angle, None
        steps = int(math.floor((2 * range_deg) / step_deg))
        rotations = [-range_deg + i * step_deg for i in range(steps + 1)]
        if 0.0 not in rotations:
            rotations.append(0.0)
        rotations = sorted(set([round(r, 3) for r in rotations]))
        return rotations, drone_angle, map_angle, None
    base = map_angle - drone_angle
    if range_deg == 0:
        return [base], drone_angle, map_angle, base
    steps = int(math.floor((2 * range_deg) / step_deg))
    rotations = [base - range_deg + i * step_deg for i in range(steps + 1)]
    if 0.0 not in rotations:
        rotations.append(0.0)
    rotations = sorted(set([round(r, 3) for r in rotations]))
    return rotations, drone_angle, map_angle, base


def _window_from_center(
    full_w: int,
    full_h: int,
    transform: Any,
    crs: Any,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    gsd_mpp: float,
) -> tuple[int, int, int, int] | None:
    if radius_m <= 0 or gsd_mpp <= 0:
        return None
    import rasterio

    xs, ys = _wgs84_to_dataset(crs, [center_lon], [center_lat])
    row, col = rasterio.transform.rowcol(transform, xs[0], ys[0])
    radius_px = int(round(radius_m / gsd_mpp))
    row0 = max(0, row - radius_px)
    row1 = min(full_h, row + radius_px)
    col0 = max(0, col - radius_px)
    col1 = min(full_w, col + radius_px)
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def _bounds_to_wgs84(crs: Any, left: float, bottom: float, right: float, top: float) -> tuple[float, float, float, float]:
    xs = [left, right, left, right]
    ys = [bottom, bottom, top, top]
    lons, lats = _dataset_to_wgs84(crs, xs, ys)
    north = max(lats)
    south = min(lats)
    east = max(lons)
    west = min(lons)
    return north, south, east, west


def _read_satellite_crop(
    sat_path: Path,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    max_size: int | None = None,
) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    window = Window(col_off, row_off, width, height)
    with rasterio.open(sat_path) as src:
        if max_size is None:
            data = src.read(window=window, out_dtype=np.uint8)
            rgb = np.transpose(data[:3], (1, 2, 0))
            return rgb
        max_dim = max(height, width)
        if max_dim <= max_size:
            data = src.read(window=window, out_dtype=np.uint8)
            rgb = np.transpose(data[:3], (1, 2, 0))
            return rgb
        scale = max_size / max_dim
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
        data = src.read(
            window=window,
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.bilinear,
            out_dtype=np.uint8,
        )
        rgb = np.transpose(data[:3], (1, 2, 0))
        return rgb


def _load_ground_truth(drone_path: Path) -> tuple[float, float] | None:
    meta = _load_drone_metadata(drone_path)
    if not meta:
        return None
    lat = meta.get("lat")
    lon = meta.get("lon")
    if lat is None or lon is None:
        return None
    return lat, lon


def _resolve_search_center(
    mode: str,
    drone_meta: dict[str, float],
    full_transform: Any,
    crs: Any,
    full_w: int,
    full_h: int,
    manual_lat: float | None,
    manual_lon: float | None,
) -> tuple[float | None, float | None, str]:
    if mode == "none":
        return None, None, "none"
    if mode == "manual":
        if manual_lat is None or manual_lon is None:
            return None, None, "manual_missing"
        return manual_lat, manual_lon, "manual"
    if mode in {"metadata", "gt"}:
        lat = drone_meta.get("lat")
        lon = drone_meta.get("lon")
        if lat is not None and lon is not None:
            return lat, lon, mode
        if mode == "metadata":
            center_col = full_w / 2.0
            center_row = full_h / 2.0
            lon_c, lat_c = _pixel_to_wgs84(full_transform, crs, center_row, center_col)
            return lat_c, lon_c, "tiff_center_fallback"
        return None, None, "gt_missing"
    if mode == "tiff_center":
        center_col = full_w / 2.0
        center_row = full_h / 2.0
        lon_c, lat_c = _pixel_to_wgs84(full_transform, crs, center_row, center_col)
        return lat_c, lon_c, "tiff_center"
    return None, None, "unknown"


def _apply_search_offset(
    lat: float | None,
    lon: float | None,
    offset_north_m: float,
    offset_east_m: float,
) -> tuple[float | None, float | None]:
    if lat is None or lon is None:
        return lat, lon
    if offset_north_m == 0.0 and offset_east_m == 0.0:
        return lat, lon
    lat_rad = math.radians(lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    if meters_per_deg_lon <= 0:
        meters_per_deg_lon = 1e-6
    lat_off = lat + (offset_north_m / meters_per_deg_lat)
    lon_off = lon + (offset_east_m / meters_per_deg_lon)
    return lat_off, lon_off


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main() -> int:
    parser = argparse.ArgumentParser(description="Coarse road-mask alignment for UAV localization")
    parser.add_argument("--config", default="configs/road_extraction.yaml")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--image", required=True, help="Path to drone image")
    parser.add_argument("--satellite", default=None, help="Path to satellite GeoTIFF")
    parser.add_argument("--sat-max-size", type=int, default=2048)
    parser.add_argument("--cache-dir", default="cache/road_masks")
    parser.add_argument("--use-osm", action="store_true")
    parser.add_argument("--osm-cache", default="cache/osm_masks")
    parser.add_argument("--osm-network-type", default="drive")
    parser.add_argument("--osm-dilate-px", type=int, default=0)
    parser.add_argument("--osm-cache-only", action="store_true")
    parser.add_argument("--osm-timeout", type=int, default=120)
    parser.add_argument("--osm-rate-limit", action="store_true", default=False)
    parser.add_argument(
        "--search-center",
        default="metadata",
        choices=["none", "gt", "metadata", "tiff_center", "manual"],
    )
    parser.add_argument("--search-radius-m", type=float, default=2000.0)
    parser.add_argument("--search-lat", type=float, default=None)
    parser.add_argument("--search-lon", type=float, default=None)
    parser.add_argument("--search-offset-north-m", type=float, default=0.0)
    parser.add_argument("--search-offset-east-m", type=float, default=0.0)
    parser.add_argument("--match-max-size", type=int, default=4096)
    parser.add_argument("--match-tolerance-px", type=int, default=3)
    parser.add_argument("--match-min-window-road-px", type=int, default=50)
    parser.add_argument("--match-f1-weight", type=float, default=10.0)
    parser.add_argument("--match-min-f1", type=float, default=0.4)
    parser.add_argument("--match-min-precision", type=float, default=0.4)
    parser.add_argument("--match-min-recall", type=float, default=0.4)
    parser.add_argument("--match-max-mean-dist-m", type=float, default=5.0)
    parser.add_argument("--match-max-p90-dist-m", type=float, default=10.0)
    parser.add_argument("--match-keypoint-tol-px", type=float, default=6.0)
    parser.add_argument("--match-min-keypoint-inlier", type=float, default=0.25)
    parser.add_argument("--match-skeletonize", action="store_true", default=True)
    parser.add_argument("--match-no-skeletonize", action="store_false", dest="match_skeletonize")
    parser.add_argument("--match-prune-iters", type=int, default=0)
    parser.add_argument("--match-close-iters", type=int, default=0)
    parser.add_argument("--match-open-iters", type=int, default=0)
    parser.add_argument("--match-kernel-size", type=int, default=3)
    parser.add_argument("--match-min-component-area", type=int, default=0)
    parser.add_argument("--reject-unless-ok", action="store_true", default=True)
    parser.add_argument("--allow-unverified", action="store_false", dest="reject_unless_ok")
    parser.add_argument("--windowed-search", action="store_true")
    parser.add_argument("--window-pad-ratio", type=float, default=0.35)
    parser.add_argument("--window-stride-ratio", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--refine-pad", type=int, default=512)
    parser.add_argument("--refine-topk", type=int, default=3)
    parser.add_argument("--refine-scale-range", type=float, default=0.15)
    parser.add_argument("--refine-scale-steps", type=int, default=5)
    parser.add_argument("--refine-rotation-range", type=float, default=10.0)
    parser.add_argument("--refine-rotation-step", type=float, default=2.0)
    parser.add_argument("--run-tag", default="latest")
    parser.add_argument("--fov-deg", type=float, default=84.0)
    parser.add_argument("--scale-range", type=float, default=0.35)
    parser.add_argument("--min-template-road-px", type=int, default=500)
    parser.add_argument("--scales", default=None, help="Comma-separated scale list for drone mask")
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=0.5)
    parser.add_argument("--scale-steps", type=int, default=7)
    parser.add_argument("--rotations", default=None)
    parser.add_argument("--rotation-range", type=float, default=30.0)
    parser.add_argument("--rotation-step", type=float, default=5.0)
    parser.add_argument("--orientation-max-size", type=int, default=2048)
    parser.add_argument("--out-dir", default="outputs/localization")
    parser.add_argument("--viz-max-size", type=int, default=2048)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--overlay-dilate", type=int, default=2)
    parser.add_argument("--overlay-color", default="255,0,0")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--qgis", action="store_true", default=True, help="Write QGIS-ready layers (default)")
    parser.add_argument("--no-qgis", dest="qgis", action="store_false", help="Disable QGIS outputs")
    parser.add_argument("--qgis-pad-px", type=int, default=0, help="Pad satellite/OSM crops (pixels)")
    parser.add_argument("--cheat-mode", action="store_true", help="Skip matching and use metadata/manual lat/lon directly")
    parser.add_argument(
        "--cheat-radius-m",
        type=float,
        default=0.0,
        help="Radius (meters) around the provided lat/lon for cheat-mode crop (<=0 uses drone footprint).",
    )
    parser.add_argument(
        "--cheat-rotation-source",
        default="phi1",
        choices=["phi1", "phi2", "none"],
        help="Yaw source for aligning the drone mask in cheat mode.",
    )
    parser.add_argument(
        "--cheat-scale-multiplier",
        type=float,
        default=1.0,
        help="Scale multiplier applied to the estimated drone footprint in cheat mode.",
    )

    parser.add_argument("--drone-cache", default="cache/road_masks")
    parser.add_argument("--no-drone-cache", action="store_true")

    parser.add_argument("--patches-per-edge", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--downscale-max-size", type=int, default=None)
    parser.add_argument("--multi-scale", type=str, default=None)
    parser.add_argument("--multi-scale-fuse", type=str, default=None)
    parser.add_argument("--sam-mask-channel", type=int, default=None)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--min-component-area", type=int, default=None)
    parser.add_argument("--close-iters", type=int, default=None)
    parser.add_argument("--open-iters", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--skeletonize", action="store_true")
    parser.add_argument("--prune-iters", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose diagnostic output")

    args = parser.parse_args()

    # Enable global verbose flag
    global _VERBOSE
    _VERBOSE = args.verbose
    if _VERBOSE:
        print("=" * 60)
        print("VERBOSE MODE ENABLED - Diagnostic logging active")
        print("=" * 60)

    cfg = _load_yaml(args.config)
    backend_name = args.backend or cfg.get("backend", "sam_road")
    backend_cfg = cfg.get(backend_name, {}) if isinstance(cfg, dict) else {}

    overrides = {
        "patches_per_edge": args.patches_per_edge,
        "batch_size": args.batch_size,
        "downscale_max_size": args.downscale_max_size,
        "multi_scale_max_sizes": _parse_int_list(args.multi_scale),
        "multi_scale_fuse": args.multi_scale_fuse,
        "mask_channel": args.sam_mask_channel,
        "auto_threshold": args.auto_threshold or None,
        "target_coverage": args.target_coverage,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
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

    backend_effective = _merge_overrides(backend_cfg, overrides)
    extractor = get_extractor(backend_name, **backend_effective)

    drone_path = Path(args.image)
    sat_path = Path(args.satellite) if args.satellite else _infer_satellite_path(drone_path)

    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform

    with rasterio.open(sat_path) as src:
        full_w = src.width
        full_h = src.height
        full_transform = src.transform
        full_crs = src.crs

    drone_meta = _load_drone_metadata(drone_path) or {}

    map_gsd_mpp, map_gsd_x, map_gsd_y = _estimate_map_gsd_mpp(
        full_transform,
        full_crs,
        full_h / 2.0,
        full_w / 2.0,
    )

    search_lat, search_lon, search_center_used = _resolve_search_center(
        args.search_center,
        drone_meta,
        full_transform,
        full_crs,
        full_w,
        full_h,
        args.search_lat,
        args.search_lon,
    )
    search_lat, search_lon = _apply_search_offset(
        search_lat,
        search_lon,
        args.search_offset_north_m,
        args.search_offset_east_m,
    )

    search_origin_row = 0
    search_origin_col = 0
    search_window = None
    if args.cheat_mode and (search_lat is None or search_lon is None):
        raise RuntimeError("Cheat mode requires a search center (metadata or manual lat/lon).")
    effective_radius_m = args.search_radius_m
    if args.cheat_mode:
        height_m = drone_meta.get("height")
        if height_m is not None and args.cheat_radius_m <= 0:
            # Estimate footprint and use half-diagonal as radius.
            drone_preview = _read_rgb(drone_path)
            drone_h, drone_w = drone_preview.shape[:2]
            drone_gsd_mpp = _estimate_drone_gsd_mpp(height_m, drone_w, drone_h, args.fov_deg)
            if drone_gsd_mpp:
                ground_w_m = drone_gsd_mpp * float(drone_w) * args.cheat_scale_multiplier
                ground_h_m = drone_gsd_mpp * float(drone_h) * args.cheat_scale_multiplier
                effective_radius_m = 0.5 * math.hypot(ground_w_m, ground_h_m)
        if args.cheat_radius_m > 0:
            effective_radius_m = args.cheat_radius_m

    if search_lat is not None and search_lon is not None and effective_radius_m and effective_radius_m > 0:
        search_window = _window_from_center(
            full_w,
            full_h,
            full_transform,
            full_crs,
            search_lat,
            search_lon,
            effective_radius_m,
            map_gsd_mpp,
        )
    if search_window:
        row0, row1, col0, col1 = search_window
        search_origin_row = row0
        search_origin_col = col0
        window_h = row1 - row0
        window_w = col1 - col0
    else:
        window_h = full_h
        window_w = full_w

    sat_rgb, sat_scale_x, sat_scale_y, sat_transform, sat_origin_row, sat_origin_col = _load_satellite_tiff(
        sat_path,
        args.sat_max_size,
        window=search_window,
    )

    # Cheat path: bypass matching and trust known coordinates. We still run segmentation for visuals.
    if args.cheat_mode:
        _log("=" * 50)
        _log("CHEAT MODE ACTIVE - using provided/metadata lat/lon, skipping matching")
        _log("=" * 50)

        # Build a reference mask for the satellite crop (OSM if requested, otherwise segment the crop).
        vis_transform = _scaled_transform(sat_transform, sat_scale_x, sat_scale_y)
        sat_mask_vis: np.ndarray
        if args.use_osm:
            from rasterio.windows import Window
            from rasterio.windows import bounds as window_bounds

            osm_cache = Path(args.osm_cache)
            osm_cache.mkdir(parents=True, exist_ok=True)
            full_cache_name = (
                f"{sat_path.stem}_osm_v2_{full_w}x{full_h}_{args.osm_network_type}_d{args.osm_dilate_px}.npy"
            )
            full_cache_path = osm_cache / full_cache_name
            sat_mask_vis = None
            if search_window:
                row0, row1, col0, col1 = search_window
                if args.osm_cache_only and full_cache_path.exists():
                    full_mask = np.load(full_cache_path, mmap_mode="r")
                    window_mask = full_mask[row0:row1, col0:col1]
                    sat_mask_vis = _downsample_mask(window_mask, (sat_rgb.shape[0], sat_rgb.shape[1]))
                    bbox = None
                else:
                    window = Window(col0, row0, col1 - col0, row1 - row0)
                    left, bottom, right, top = window_bounds(window, full_transform)
                    bbox = _bounds_to_wgs84(full_crs, left, bottom, right, top)
            else:
                if args.osm_cache_only and full_cache_path.exists():
                    full_mask = np.load(full_cache_path, mmap_mode="r")
                    sat_mask_vis = _downsample_mask(full_mask, (sat_rgb.shape[0], sat_rgb.shape[1]))
                    bbox = None
                else:
                    bbox = None
            vis_key = (
                f"{sat_path.stem}_cheat_{search_origin_row}_{search_origin_col}_{sat_rgb.shape[1]}x{sat_rgb.shape[0]}"
                if search_window
                else f"{sat_path.stem}_cheat_full"
            )
            if sat_mask_vis is None:
                sat_mask_vis = _build_osm_mask(
                    sat_path,
                    out_shape=(sat_rgb.shape[0], sat_rgb.shape[1]),
                    transform=vis_transform,
                    dst_crs=full_crs,
                    cache_dir=osm_cache,
                    network_type=args.osm_network_type,
                    osm_dilate_px=args.osm_dilate_px,
                    bbox=bbox,
                    cache_key=vis_key,
                    cache_only=args.osm_cache_only,
                    osm_timeout=args.osm_timeout,
                    osm_rate_limit=args.osm_rate_limit,
                )
        else:
            sat_mask_vis = extractor.predict_mask(sat_rgb)

        # Drone road mask (use cache if available).
        _log("Running road extraction on drone image (cheat mode)...")
        drone_rgb = _read_rgb(drone_path)
        drone_mask = None
        drone_cache_path = None
        if not args.no_drone_cache:
            drone_cache = Path(args.drone_cache)
            drone_cache.mkdir(parents=True, exist_ok=True)
            backend_sig = json.dumps(backend_effective, sort_keys=True)
            cache_hash = hashlib.md5(backend_sig.encode("utf-8")).hexdigest()[:8]
            drone_cache_path = drone_cache / f"{drone_path.stem}_{backend_name}_{cache_hash}_mask.npy"
            if drone_cache_path.exists():
                drone_mask = np.load(drone_cache_path)
                _log(f"Loaded drone mask from cache: {drone_cache_path}")
        if drone_mask is None:
            t0 = time.time()
            drone_mask = extractor.predict_mask(drone_rgb)
            t1 = time.time()
            _log(f"Road extraction took {t1-t0:.2f}s")
            if drone_cache_path is not None:
                np.save(drone_cache_path, drone_mask)

        # Scale the drone mask to the estimated ground footprint, then place at the target center.
        drone_h, drone_w = drone_mask.shape[:2]
        height_m = drone_meta.get("height")
        drone_gsd_mpp = None
        if height_m is not None:
            drone_gsd_mpp = _estimate_drone_gsd_mpp(height_m, drone_w, drone_h, args.fov_deg)
        full_w_px = None
        full_h_px = None
        if drone_gsd_mpp and map_gsd_x > 0 and map_gsd_y > 0:
            ground_w_m = drone_gsd_mpp * float(drone_w) * args.cheat_scale_multiplier
            ground_h_m = drone_gsd_mpp * float(drone_h) * args.cheat_scale_multiplier
            full_w_px = max(2.0, ground_w_m / map_gsd_x)
            full_h_px = max(2.0, ground_h_m / map_gsd_y)
            vis_w = max(2, int(round(full_w_px / max(1e-6, sat_scale_x))))
            vis_h = max(2, int(round(full_h_px / max(1e-6, sat_scale_y))))
        else:
            # Fallback: keep the old behavior if height is missing.
            vis_h, vis_w = sat_rgb.shape[:2]

        # Apply yaw if available for alignment to north-up satellite.
        yaw_deg = 0.0
        if args.cheat_rotation_source == "phi1":
            yaw_deg = float(drone_meta.get("phi1", 0.0))
        elif args.cheat_rotation_source == "phi2":
            yaw_deg = float(drone_meta.get("phi2", 0.0))
        rot_mask = _rotate_mask((drone_mask > 0).astype(np.uint8), -yaw_deg)
        vis_drone_mask = _resize_mask_to_shape(rot_mask, (vis_h, vis_w))
        overlay_mask = np.zeros(sat_rgb.shape[:2], dtype=np.uint8)

        import rasterio

        xs, ys = _wgs84_to_dataset(full_crs, [search_lon], [search_lat])
        row, col = rasterio.transform.rowcol(full_transform, xs[0], ys[0])
        vis_cx = (col - sat_origin_col) / max(1e-6, sat_scale_x)
        vis_cy = (row - sat_origin_row) / max(1e-6, sat_scale_y)
        x0 = int(round(vis_cx - vis_w / 2.0))
        y0 = int(round(vis_cy - vis_h / 2.0))
        x1 = x0 + vis_w
        y1 = y0 + vis_h
        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(sat_rgb.shape[1], x1)
        dst_y1 = min(sat_rgb.shape[0], y1)
        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            src_x1 = src_x0 + (dst_x1 - dst_x0)
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            overlay_mask[dst_y0:dst_y1, dst_x0:dst_x1] = vis_drone_mask[src_y0:src_y1, src_x0:src_x1] * 255

        color = tuple(int(x) for x in args.overlay_color.split(",") if x.strip())
        overlay_drone = _overlay_mask(
            sat_rgb,
            overlay_mask,
            color=color,
            alpha=args.overlay_alpha,
            dilate_iters=args.overlay_dilate,
        )
        overlay_ref = _overlay_mask(
            sat_rgb,
            sat_mask_vis * 255,
            color=(0, 255, 0),
            alpha=args.overlay_alpha,
            dilate_iters=args.overlay_dilate,
        )
        overlay_both = _overlay_mask(
            overlay_ref,
            overlay_mask,
            color=color,
            alpha=args.overlay_alpha,
            dilate_iters=args.overlay_dilate,
        )

        out_dir = Path(args.out_dir) / drone_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_image(out_dir / "sat_overlay_drone_cheat.png", overlay_drone)
        _save_image(out_dir / "sat_overlay_ref_cheat.png", overlay_ref)
        _save_image(out_dir / "sat_overlay_both_cheat.png", overlay_both)
        _save_image(out_dir / "sat_crop_cheat.png", sat_rgb)
        _save_image(out_dir / "drone_raw.png", drone_rgb)
        drone_overlay = _overlay_mask(
            drone_rgb,
            (drone_mask > 0).astype(np.uint8) * 255,
            color=color,
            alpha=args.overlay_alpha,
            dilate_iters=args.overlay_dilate,
        )
        _save_image(out_dir / "drone_overlay.png", drone_overlay)
        _save_image(out_dir / "drone_mask.png", np.stack([(drone_mask > 0).astype(np.uint8) * 255] * 3, axis=2))

        report = (
            f"drone_image={drone_path}\n"
            f"satellite={sat_path}\n"
            f"backend={backend_name}\n"
            f"backend_config={json.dumps(backend_effective, sort_keys=True)}\n"
            f"cheat_mode=True\n"
            f"lat={search_lat}\n"
            f"lon={search_lon}\n"
            f"pix_row={row}\n"
            f"pix_col={col}\n"
            f"overlay_size_vis=({vis_w}, {vis_h})\n"
            f"drone_gsd_mpp={drone_gsd_mpp}\n"
            f"map_gsd_x={map_gsd_x}\n"
            f"map_gsd_y={map_gsd_y}\n"
            f"cheat_yaw_deg={yaw_deg}\n"
            f"cheat_radius_m={effective_radius_m}\n"
            f"search_center_used={search_center_used}\n"
            f"search_radius_m={effective_radius_m}\n"
            f"sat_window={search_window}\n"
            f"sat_max_size={args.sat_max_size}\n"
            f"use_osm={args.use_osm}\n"
        )
        gt = _load_ground_truth(drone_path)
        if gt is not None:
            error_km = _haversine_km(search_lat, search_lon, gt[0], gt[1])
            report += f"gt_lat={gt[0]}\n"
            report += f"gt_lon={gt[1]}\n"
            report += f"error_km={error_km}\n"
        (out_dir / "result.txt").write_text(report, encoding="utf-8")

        if args.qgis:
            scale_full_x = None
            scale_full_y = None
            if full_w_px is not None and full_h_px is not None:
                scale_full_x = full_w_px / float(drone_w)
                scale_full_y = full_h_px / float(drone_h)
            full_osm_mask = None
            if args.use_osm:
                full_cache_name = (
                    f"{sat_path.stem}_osm_v2_{full_w}x{full_h}_{args.osm_network_type}_d{args.osm_dilate_px}.npy"
                )
                full_cache_path = Path(args.osm_cache) / full_cache_name
                if full_cache_path.exists():
                    full_osm_mask = np.load(full_cache_path, mmap_mode="r")
            _export_qgis_outputs(
                out_dir=out_dir,
                drone_rgb=drone_rgb,
                drone_mask=drone_mask,
                sat_path=sat_path,
                full_transform=full_transform,
                full_crs=full_crs,
                full_w=full_w,
                full_h=full_h,
                center_row=float(row),
                center_col=float(col),
                scale_full_x=scale_full_x,
                scale_full_y=scale_full_y,
                rotation_deg=-yaw_deg,
                lat=search_lat,
                lon=search_lon,
                use_osm=args.use_osm,
                osm_cache_dir=Path(args.osm_cache),
                osm_network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                osm_cache_only=args.osm_cache_only,
                osm_timeout=args.osm_timeout,
                osm_rate_limit=args.osm_rate_limit,
                full_osm_mask=full_osm_mask,
                pad_px=args.qgis_pad_px,
            )
        return 0

    match_mask = None
    full_mask = None
    match_stride = 1
    match_origin_row = search_origin_row
    match_origin_col = search_origin_col

    _log("=" * 50)
    _log("LOADING OSM / SATELLITE MASK")
    _log("=" * 50)
    _log(f"search_window={search_window}")
    _log(f"window_h={window_h}, window_w={window_w}")

    if args.use_osm:
        osm_cache = Path(args.osm_cache)
        osm_cache.mkdir(parents=True, exist_ok=True)

        window_mask = None
        max_dim = max(window_h, window_w)
        if args.match_max_size and args.match_max_size > 0:
            match_stride = max(1, int(math.ceil(max_dim / args.match_max_size)))
        match_h = int(math.ceil(window_h / match_stride))
        match_w = int(math.ceil(window_w / match_stride))
        
        _log(f"use_osm=True, match_stride={match_stride}, match_h={match_h}, match_w={match_w}")

        full_cache_name = (
            f"{sat_path.stem}_osm_v2_{full_w}x{full_h}_{args.osm_network_type}_d{args.osm_dilate_px}.npy"
        )
        full_cache_path = osm_cache / full_cache_name
        _log(f"Looking for full cache at: {full_cache_path}")
        if full_cache_path.exists():
            _log("Found full OSM cache, loading...")
            full_mask = np.load(full_cache_path, mmap_mode="r")
            _log(f"full_mask.shape={full_mask.shape}, road_px={int(np.count_nonzero(full_mask > 0))}")
            if search_window:
                window_mask = full_mask[
                    match_origin_row : match_origin_row + window_h,
                    match_origin_col : match_origin_col + window_w,
                ]
                _log(f"Extracted window_mask from [{match_origin_row}:{match_origin_row + window_h}, {match_origin_col}:{match_origin_col + window_w}]")
            else:
                window_mask = full_mask
            _log(f"window_mask.shape={window_mask.shape}, road_px={int(np.count_nonzero(window_mask > 0))}")
            if match_stride == 1:
                match_mask = np.array(window_mask, dtype=np.uint8)
            else:
                match_mask = _downsample_mask(window_mask, (match_h, match_w))
            match_h, match_w = match_mask.shape[:2]
            _log(f"match_mask.shape={match_mask.shape}, road_px={int(np.count_nonzero(match_mask > 0))}")
        else:
            _log("Full cache not found, building OSM mask...")
            if search_window:
                window = Window(match_origin_col, match_origin_row, window_w, window_h)
                window_tr = window_transform(window, full_transform)
                left, bottom, right, top = window_bounds(window, full_transform)
                bbox = _bounds_to_wgs84(full_crs, left, bottom, right, top)
            else:
                window_tr = full_transform
                bbox = None
            if match_stride == 1:
                match_transform = window_tr
            else:
                match_transform = _scaled_transform(window_tr, match_stride, match_stride)
            cache_key = (
                f"{sat_path.stem}_match_{match_origin_row}_{match_origin_col}_{match_w}x{match_h}_s{match_stride}"
            )
            match_mask = _build_osm_mask(
                sat_path,
                out_shape=(match_h, match_w),
                transform=match_transform,
                dst_crs=full_crs,
                cache_dir=osm_cache,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=bbox,
                cache_key=cache_key,
                cache_only=args.osm_cache_only,
                osm_timeout=args.osm_timeout,
                osm_rate_limit=args.osm_rate_limit,
            )

        vis_transform = _scaled_transform(sat_transform, sat_scale_x, sat_scale_y)
        vis_bbox = None
        if search_window:
            left, bottom, right, top = window_bounds(
                Window(search_origin_col, search_origin_row, window_w, window_h),
                full_transform,
            )
            vis_bbox = _bounds_to_wgs84(full_crs, left, bottom, right, top)
        vis_key = (
            f"{sat_path.stem}_vis_{search_origin_row}_{search_origin_col}_{window_w}x{window_h}"
            if search_window
            else f"{sat_path.stem}_vis"
        )
        if args.osm_cache_only and window_mask is not None:
            sat_mask_vis = _downsample_mask(
                window_mask, (sat_rgb.shape[0], sat_rgb.shape[1])
            )
        elif args.osm_cache_only and match_mask is not None:
            sat_mask_vis = _downsample_mask(
                match_mask, (sat_rgb.shape[0], sat_rgb.shape[1])
            )
        else:
            sat_mask_vis = _build_osm_mask(
                sat_path,
                out_shape=(sat_rgb.shape[0], sat_rgb.shape[1]),
                transform=vis_transform,
                dst_crs=full_crs,
                cache_dir=osm_cache,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=vis_bbox,
                cache_key=vis_key,
                cache_only=args.osm_cache_only,
                osm_timeout=args.osm_timeout,
                osm_rate_limit=args.osm_rate_limit,
            )
    else:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sat_id = sat_path.stem
        if search_window:
            sat_mask_path = cache_dir / (
                f"{sat_id}_mask_{search_origin_row}_{search_origin_col}_{window_w}x{window_h}_{args.sat_max_size}.npy"
            )
        else:
            sat_mask_path = cache_dir / f"{sat_id}_mask_{args.sat_max_size}.npy"

        if sat_mask_path.exists():
            sat_mask = np.load(sat_mask_path)
        else:
            sat_mask = extractor.predict_mask(sat_rgb)
            np.save(sat_mask_path, sat_mask)

        match_mask = (sat_mask > 0).astype(np.uint8)
        sat_mask_vis = match_mask
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for distance transform") from exc
    if match_mask is None:
        raise RuntimeError("Failed to build a match mask")
    
    _log("=" * 50)
    _log("PREPARING MAP MASK")
    _log("=" * 50)
    _log(f"match_mask.shape={match_mask.shape}, road_px={int(np.count_nonzero(match_mask > 0))}")
    _log(f"match_skeletonize={args.match_skeletonize}, match_prune_iters={args.match_prune_iters}")
    
    map_match = _prepare_match_mask(
        match_mask,
        close_iters=args.match_close_iters,
        open_iters=args.match_open_iters,
        kernel_size=args.match_kernel_size,
        min_component_area=args.match_min_component_area,
        skeletonize=args.match_skeletonize,
        prune_iters=args.match_prune_iters,
    )
    _log(f"map_match after prepare: road_px={int(np.count_nonzero(map_match > 0))}")
    
    dist_map = _distance_transform(map_match).astype(np.float32)
    _log(f"dist_map.shape={dist_map.shape}, min={dist_map.min():.2f}, max={dist_map.max():.2f}")

    _log("=" * 50)
    _log("PREPARING DRONE MASK")
    _log("=" * 50)
    
    drone_rgb = _read_rgb(drone_path)
    _log(f"drone_rgb.shape={drone_rgb.shape}")
    
    drone_mask = None
    drone_cache_path = None
    if not args.no_drone_cache:
        drone_cache = Path(args.drone_cache)
        drone_cache.mkdir(parents=True, exist_ok=True)
        backend_sig = json.dumps(backend_effective, sort_keys=True)
        cache_hash = hashlib.md5(backend_sig.encode("utf-8")).hexdigest()[:8]
        drone_cache_path = drone_cache / f"{drone_path.stem}_{backend_name}_{cache_hash}_mask.npy"
        if drone_cache_path.exists():
            drone_mask = np.load(drone_cache_path)
            _log(f"Loaded drone mask from cache: {drone_cache_path}")
    if drone_mask is None:
        _log("Running road extraction on drone image...")
        t0 = time.time()
        drone_mask = extractor.predict_mask(drone_rgb)
        t1 = time.time()
        _log(f"Road extraction took {t1-t0:.2f}s")
        if drone_cache_path is not None:
            np.save(drone_cache_path, drone_mask)
    
    _log(f"drone_mask.shape={drone_mask.shape}, road_px={int(np.count_nonzero(drone_mask > 0))}")
    
    drone_match = _prepare_match_mask(
        drone_mask,
        close_iters=args.match_close_iters,
        open_iters=args.match_open_iters,
        kernel_size=args.match_kernel_size,
        min_component_area=args.match_min_component_area,
        skeletonize=args.match_skeletonize,
        prune_iters=args.match_prune_iters,
    )
    _log(f"drone_match after prepare: road_px={int(np.count_nonzero(drone_match > 0))}")

    drone_gsd_mpp = None
    scale_est = None
    height_m = drone_meta.get("height")
    if height_m is not None:
        drone_gsd_mpp = _estimate_drone_gsd_mpp(
            height_m,
            drone_rgb.shape[1],
            drone_rgb.shape[0],
            args.fov_deg,
        )
    if drone_gsd_mpp and map_gsd_mpp > 0:
        scale_est = drone_gsd_mpp / map_gsd_mpp

    scales_full = _parse_float_list(args.scales)
    if scales_full is None:
        if scale_est is not None:
            scale_min = max(0.02, scale_est * (1.0 - args.scale_range))
            scale_max = min(2.0, scale_est * (1.0 + args.scale_range))
            if args.scale_steps <= 1:
                scales_full = [scale_est]
            else:
                scales_full = list(np.linspace(scale_min, scale_max, args.scale_steps))
        else:
            if args.scale_steps <= 1:
                scales_full = [args.scale_min]
            else:
                scales_full = list(np.linspace(args.scale_min, args.scale_max, args.scale_steps))
    elif scale_est is not None and args.scale_range > 0:
        scale_min = max(0.01, scale_est * (1.0 - args.scale_range))
        scale_max = scale_est * (1.0 + args.scale_range)
        filtered = [s for s in scales_full if scale_min <= s <= scale_max]
        if filtered:
            scales_full = filtered
        else:
            scales_full = [scale_est]

    rotations = _parse_float_list(args.rotations)
    rot_drone_angle = None
    rot_map_angle = None
    rot_base = None
    if rotations is None:
        rotations, rot_drone_angle, rot_map_angle, rot_base = _auto_rotation_list(
            drone_match,
            map_match,
            args.rotation_range,
            args.rotation_step,
            args.orientation_max_size,
        )
    if not rotations:
        rotations = [0.0]

    scales_match = [s / float(match_stride) for s in scales_full]
    min_road_px_match = args.min_template_road_px
    if match_stride > 1 and args.min_template_road_px:
        if args.match_skeletonize:
            min_road_px_match = max(1, int(round(args.min_template_road_px / float(match_stride))))
        else:
            min_road_px_match = max(1, int(round(args.min_template_road_px / float(match_stride * match_stride))))
    min_window_road_px_match = args.match_min_window_road_px
    if match_stride > 1 and args.match_min_window_road_px:
        if args.match_skeletonize:
            min_window_road_px_match = max(1, int(round(args.match_min_window_road_px / float(match_stride))))
        else:
            min_window_road_px_match = max(
                1, int(round(args.match_min_window_road_px / float(match_stride * match_stride)))
            )
    if args.match_tolerance_px <= 0:
        tol_px_match = 0
    elif match_stride > 1:
        tol_px_match = max(1, int(round(args.match_tolerance_px / float(match_stride))))
    else:
        tol_px_match = args.match_tolerance_px
    base_scale = (scale_est / float(match_stride)) if scale_est is not None else None
    
    _log("=" * 50)
    _log("COARSE MATCHING PHASE")
    _log("=" * 50)
    _log(f"scale_est={scale_est}, match_stride={match_stride}")
    _log(f"scales_full={scales_full}")
    _log(f"scales_match={scales_match}")
    _log(f"rotations={rotations}")
    _log(f"min_road_px_match={min_road_px_match}, min_window_road_px_match={min_window_road_px_match}")
    _log(f"base_scale={base_scale}")
    _log(f"dist_map.shape={dist_map.shape}, map_match.shape={map_match.shape}")
    _log(f"drone_match.shape={drone_match.shape}")
    _log(f"drone_match road pixels={int(np.count_nonzero(drone_match > 0))}")
    _log(f"map_match road pixels={int(np.count_nonzero(map_match > 0))}")
    _log(f"windowed_search={args.windowed_search}")
    
    t0 = time.time()
    if args.windowed_search:
        coarse_matches = _windowed_match_candidates(
            dist_map,
            map_match,
            drone_match,
            scales_match,
            rotations,
            args.top_k,
            min_road_px_match,
            min_window_road_px_match,
            args.window_pad_ratio,
            args.window_stride_ratio,
            base_scale,
        )
    else:
        coarse_matches = _match_candidates(
            dist_map,
            map_match,
            drone_match,
            scales_match,
            rotations,
            args.top_k,
            min_road_px_match,
            min_window_road_px_match,
        )
    t1 = time.time()
    _log(f"Coarse matching took {t1-t0:.2f}s, found {len(coarse_matches)} matches before scoring")
    
    coarse_matches = _score_candidates(coarse_matches, map_match, tol_px_match, args.match_f1_weight)
    _log(f"After scoring: {len(coarse_matches)} matches")
    if coarse_matches:
        _log(f"Best match: score={coarse_matches[0].get('score')}, scale={coarse_matches[0].get('scale')}, rotation={coarse_matches[0].get('rotation')}")
    
    if not coarse_matches:
        _log("ERROR: No valid coarse match found!")
        _log("Possible causes:")
        _log("  1. All scales result in template smaller than min_template_road_px")
        _log("  2. All rotated templates are larger than the search window")
        _log("  3. No map positions have >= min_window_road_px road pixels")
        _log("  4. drone_match has 0 road pixels (segmentation failed)")
        _log("  5. map_match has 0 road pixels (OSM area is empty)")
        raise RuntimeError("No valid coarse match found; adjust scale/rotation range")

    best = coarse_matches[0]
    best_origin_row = match_origin_row
    best_origin_col = match_origin_col
    best_mask_map = match_mask
    best_match_map = map_match
    best_scale_to_full = match_stride
    refine_matches: list[dict[str, Any]] = []
    best_dist_map = dist_map

    if args.use_osm and (match_stride > 1 or args.refine_scale_range > 0 or args.refine_rotation_range > 0):
        candidate_pool = coarse_matches[: max(1, int(args.refine_topk))]
        refined_candidates: list[dict[str, Any]] = []
        for candidate in candidate_pool:
            base_scale_full = float(candidate["scale"]) * match_stride
            base_rotation = float(candidate["rotation"])
            coarse_off_x, coarse_off_y = candidate["offset"]
            full_off_x = match_origin_col + int(round(coarse_off_x * match_stride))
            full_off_y = match_origin_row + int(round(coarse_off_y * match_stride))
            tmpl_w_full = max(1, int(round(candidate["mask"].shape[1] * match_stride)))
            tmpl_h_full = max(1, int(round(candidate["mask"].shape[0] * match_stride)))

            pad = max(0, int(args.refine_pad))
            row0 = max(0, full_off_y - pad)
            row1 = min(full_h, full_off_y + tmpl_h_full + pad)
            col0 = max(0, full_off_x - pad)
            col1 = min(full_w, full_off_x + tmpl_w_full + pad)
            ref_h = max(1, row1 - row0)
            ref_w = max(1, col1 - col0)

            if full_mask is not None:
                ref_mask = np.array(full_mask[row0:row1, col0:col1], dtype=np.uint8)
            else:
                window = Window(col0, row0, ref_w, ref_h)
                window_tr = window_transform(window, full_transform)
                left, bottom, right, top = window_bounds(window, full_transform)
                bbox = _bounds_to_wgs84(full_crs, left, bottom, right, top)
                cache_key = f"{sat_path.stem}_ref_{row0}_{col0}_{ref_w}x{ref_h}"
                ref_mask = _build_osm_mask(
                    sat_path,
                    out_shape=(ref_h, ref_w),
                    transform=window_tr,
                    dst_crs=full_crs,
                    cache_dir=Path(args.osm_cache),
                    network_type=args.osm_network_type,
                    osm_dilate_px=args.osm_dilate_px,
                    bbox=bbox,
                    cache_key=cache_key,
                    cache_only=args.osm_cache_only,
                    osm_timeout=args.osm_timeout,
                    osm_rate_limit=args.osm_rate_limit,
                )

            ref_match = _prepare_match_mask(
                ref_mask,
                close_iters=args.match_close_iters,
                open_iters=args.match_open_iters,
                kernel_size=args.match_kernel_size,
                min_component_area=args.match_min_component_area,
                skeletonize=args.match_skeletonize,
                prune_iters=args.match_prune_iters,
            )
            dist_map_ref = _distance_transform(ref_match).astype(np.float32)
            refine_scales = _scale_list_around(base_scale_full, args.refine_scale_range, args.refine_scale_steps)
            if args.refine_rotation_range <= 0 or args.refine_rotation_step <= 0:
                refine_rotations = [base_rotation]
            else:
                steps = int(math.floor((2 * args.refine_rotation_range) / args.refine_rotation_step))
                refine_rotations = [
                    base_rotation - args.refine_rotation_range + i * args.refine_rotation_step
                    for i in range(steps + 1)
                ]

            refine_matches_local = _match_candidates(
                dist_map_ref,
                ref_match,
                drone_match,
                refine_scales,
                refine_rotations,
                args.top_k,
                args.min_template_road_px,
                args.match_min_window_road_px,
            )
            refine_matches_local = _score_candidates(
                refine_matches_local, ref_match, args.match_tolerance_px, args.match_f1_weight
            )
            if refine_matches_local:
                refined_best = refine_matches_local[0]
                refined_candidates.append(
                    {
                        "best": refined_best,
                        "origin_row": row0,
                        "origin_col": col0,
                        "mask_map": ref_mask,
                        "match_map": ref_match,
                        "scale_to_full": 1,
                        "dist_map": dist_map_ref,
                        "refine_matches": refine_matches_local,
                    }
                )
            else:
                refined_candidates.append(
                    {
                        "best": candidate,
                        "origin_row": match_origin_row,
                        "origin_col": match_origin_col,
                        "mask_map": match_mask,
                        "match_map": map_match,
                        "scale_to_full": match_stride,
                        "dist_map": dist_map,
                        "refine_matches": [],
                    }
                )

        refined_candidates.sort(
            key=lambda item: float(item["best"].get("score_combined", item["best"].get("score", 0.0)))
        )
        best_info = refined_candidates[0]
        best = best_info["best"]
        best_origin_row = best_info["origin_row"]
        best_origin_col = best_info["origin_col"]
        best_mask_map = best_info["mask_map"]
        best_match_map = best_info["match_map"]
        best_scale_to_full = best_info["scale_to_full"]
        best_dist_map = best_info["dist_map"]
        refine_matches = best_info["refine_matches"]

    best_mask = best["mask"]
    best_mask_raw = _resize_mask(drone_mask, float(best["scale"]))
    if best_mask_raw is not None:
        best_mask_raw = _rotate_mask(best_mask_raw, float(best["rotation"]))
    else:
        best_mask_raw = best_mask
    best_mask_vis = best_mask_raw
    off_x, off_y = best["offset"]

    if args.use_osm:
        scale_to_full_x = best_scale_to_full
        scale_to_full_y = best_scale_to_full
    else:
        scale_to_full_x = sat_scale_x
        scale_to_full_y = sat_scale_y

    full_off_x = best_origin_col + int(round(off_x * scale_to_full_x))
    full_off_y = best_origin_row + int(round(off_y * scale_to_full_y))

    vis_off_x = int(round((full_off_x - sat_origin_col) / sat_scale_x))
    vis_off_y = int(round((full_off_y - sat_origin_row) / sat_scale_y))
    vis_mask_h = max(1, int(round(best_mask_vis.shape[0] * scale_to_full_y / sat_scale_y)))
    vis_mask_w = max(1, int(round(best_mask_vis.shape[1] * scale_to_full_x / sat_scale_x)))
    if scale_to_full_x == 1 and scale_to_full_y == 1 and sat_scale_x == 1.0 and sat_scale_y == 1.0:
        vis_mask = best_mask_vis
    else:
        vis_mask = _resize_mask_to_shape(best_mask_vis, (vis_mask_h, vis_mask_w))

    center_x_full = full_off_x + (best_mask_vis.shape[1] * scale_to_full_x) / 2.0
    center_y_full = full_off_y + (best_mask_vis.shape[0] * scale_to_full_y) / 2.0

    orig_col = center_x_full
    orig_row = center_y_full

    lon, lat = _pixel_to_wgs84(full_transform, full_crs, orig_row, orig_col)

    out_dir = Path(args.out_dir) / drone_path.stem
    if args.run_tag and args.run_tag != "latest":
        if args.run_tag == "auto":
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            tag = args.run_tag
        out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_candidates_csv(out_dir / "candidates_coarse.csv", coarse_matches)
    if refine_matches:
        _write_candidates_csv(out_dir / "candidates_refine.csv", refine_matches)

    overlay_mask = np.zeros(sat_rgb.shape[:2], dtype=np.uint8)
    h, w = vis_mask.shape[:2]
    max_x = min(vis_off_x + w, sat_rgb.shape[1])
    max_y = min(vis_off_y + h, sat_rgb.shape[0])
    mask_w = max_x - vis_off_x
    mask_h = max_y - vis_off_y
    if mask_w > 0 and mask_h > 0:
        overlay_mask[vis_off_y:max_y, vis_off_x:max_x] = (
            (vis_mask[:mask_h, :mask_w] > 0).astype(np.uint8) * 255
        )
    color = tuple(int(x) for x in args.overlay_color.split(",") if x.strip())
    overlay_img = _overlay_mask(
        sat_rgb,
        overlay_mask,
        color=color,
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / "sat_overlay.png", overlay_img)
    context_img = _draw_rect(
        sat_rgb,
        vis_off_x,
        vis_off_y,
        vis_off_x + max(0, mask_w - 1),
        vis_off_y + max(0, mask_h - 1),
        color=color,
        thickness=2,
    )
    _save_image(out_dir / "match_context.png", context_img)

    drone_overlay = _overlay_mask(
        drone_rgb,
        (drone_mask > 0).astype(np.uint8) * 255,
        color=color,
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / "drone_overlay.png", drone_overlay)

    drone_mask_rgb = np.stack([(drone_mask > 0).astype(np.uint8) * 255] * 3, axis=2)
    _save_image(out_dir / "drone_mask.png", drone_mask_rgb)

    drone_match_rgb = np.stack([drone_match.astype(np.uint8) * 255] * 3, axis=2)
    _save_image(out_dir / "drone_match_mask.png", drone_match_rgb)

    sat_mask_rgb = np.stack([sat_mask_vis * 255] * 3, axis=2)
    _save_image(out_dir / "sat_mask.png", sat_mask_rgb)

    map_match_rgb = np.stack([map_match.astype(np.uint8) * 255] * 3, axis=2)
    _save_image(out_dir / "map_match_mask.png", map_match_rgb)

    target_h = drone_rgb.shape[0]
    panels = [
        _resize_to_height(drone_rgb, target_h),
        _resize_to_height(drone_mask_rgb, target_h),
        _resize_to_height(drone_overlay, target_h),
        _resize_to_height(overlay_img, target_h),
    ]
    comp = np.concatenate(panels, axis=1)
    _save_image(out_dir / "comparison.png", comp)

    ref_label = "osm" if args.use_osm else "satmask"
    crop_h = max(0, mask_h)
    crop_w = max(0, mask_w)
    if crop_h == 0 or crop_w == 0:
        raise RuntimeError("Matched crop falls outside satellite bounds")

    sat_crop = sat_rgb[vis_off_y : vis_off_y + crop_h, vis_off_x : vis_off_x + crop_w]
    sat_mask_crop = sat_mask_vis[vis_off_y : vis_off_y + crop_h, vis_off_x : vis_off_x + crop_w]
    vis_mask_crop = vis_mask[:crop_h, :crop_w]

    drone_scaled = _resize_to_shape(drone_rgb, (crop_h, crop_w))
    angle = float(best["rotation"] or 0.0)
    if angle != 0.0:
        drone_scaled = _rotate_image(drone_scaled, angle)

    _save_image(out_dir / "sat_crop.png", sat_crop)
    _save_image(out_dir / "drone_scaled.png", drone_scaled)
    match_pair = np.concatenate([drone_scaled, sat_crop], axis=1)
    _save_image(out_dir / "match_pair.png", match_pair)
    _save_image(out_dir / "match_blend.png", _blend_images(drone_scaled, sat_crop, alpha=0.5))

    sat_crop_overlay_drone = _overlay_mask(
        sat_crop,
        (vis_mask_crop > 0).astype(np.uint8) * 255,
        color=color,
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / "sat_crop_overlay_drone.png", sat_crop_overlay_drone)

    sat_crop_overlay_ref = _overlay_mask(
        sat_crop,
        sat_mask_crop * 255,
        color=(0, 255, 0),
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / f"sat_crop_overlay_{ref_label}.png", sat_crop_overlay_ref)

    sat_crop_overlay_both = _overlay_mask(
        sat_crop_overlay_ref,
        (vis_mask_crop > 0).astype(np.uint8) * 255,
        color=color,
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / "sat_crop_overlay_both.png", sat_crop_overlay_both)

    drone_scaled_overlay_ref = _overlay_mask(
        drone_scaled,
        sat_mask_crop * 255,
        color=(0, 255, 0),
        alpha=args.overlay_alpha,
        dilate_iters=args.overlay_dilate,
    )
    _save_image(out_dir / f"drone_scaled_overlay_{ref_label}.png", drone_scaled_overlay_ref)

    if args.use_osm and best_scale_to_full == 1:
        full_mask_h, full_mask_w = best_mask_vis.shape[:2]
        full_x0 = max(0, full_off_x)
        full_y0 = max(0, full_off_y)
        full_x1 = min(full_w, full_off_x + full_mask_w)
        full_y1 = min(full_h, full_off_y + full_mask_h)
        full_crop_w = full_x1 - full_x0
        full_crop_h = full_y1 - full_y0
        if full_crop_w > 0 and full_crop_h > 0:
            mask_x0 = full_x0 - full_off_x
            mask_y0 = full_y0 - full_off_y
            mask_crop_full = best_mask_vis[
                mask_y0 : mask_y0 + full_crop_h,
                mask_x0 : mask_x0 + full_crop_w,
            ]
            ref_x0 = full_x0 - best_origin_col
            ref_y0 = full_y0 - best_origin_row
            ref_crop_full = best_mask_map[
                ref_y0 : ref_y0 + full_crop_h,
                ref_x0 : ref_x0 + full_crop_w,
            ]

            sat_crop_full = _read_satellite_crop(
                sat_path,
                full_x0,
                full_y0,
                full_crop_w,
                full_crop_h,
                max_size=args.viz_max_size,
            )
            mask_crop_vis = _resize_mask_to_shape(mask_crop_full, sat_crop_full.shape[:2])
            ref_crop_vis = _resize_mask_to_shape(ref_crop_full, sat_crop_full.shape[:2])
            drone_scaled_full = _resize_to_shape(drone_rgb, sat_crop_full.shape[:2])
            if angle != 0.0:
                drone_scaled_full = _rotate_image(drone_scaled_full, angle)

            _save_image(out_dir / "sat_crop_full.png", sat_crop_full)
            _save_image(out_dir / "drone_scaled_full.png", drone_scaled_full)
            match_pair_full = np.concatenate([drone_scaled_full, sat_crop_full], axis=1)
            _save_image(out_dir / "match_pair_full.png", match_pair_full)
            _save_image(
                out_dir / "match_blend_full.png",
                _blend_images(drone_scaled_full, sat_crop_full, alpha=0.5),
            )

            sat_crop_full_overlay_drone = _overlay_mask(
                sat_crop_full,
                (mask_crop_vis > 0).astype(np.uint8) * 255,
                color=color,
                alpha=args.overlay_alpha,
                dilate_iters=args.overlay_dilate,
            )
            _save_image(out_dir / "sat_crop_full_overlay_drone.png", sat_crop_full_overlay_drone)

            sat_crop_full_overlay_ref = _overlay_mask(
                sat_crop_full,
                ref_crop_vis * 255,
                color=(0, 255, 0),
                alpha=args.overlay_alpha,
                dilate_iters=args.overlay_dilate,
            )
            _save_image(
                out_dir / f"sat_crop_full_overlay_{ref_label}.png",
                sat_crop_full_overlay_ref,
            )

            sat_crop_full_overlay_both = _overlay_mask(
                sat_crop_full_overlay_ref,
                (mask_crop_vis > 0).astype(np.uint8) * 255,
                color=color,
                alpha=args.overlay_alpha,
                dilate_iters=args.overlay_dilate,
            )
            _save_image(out_dir / "sat_crop_full_overlay_both.png", sat_crop_full_overlay_both)

            drone_scaled_full_overlay_ref = _overlay_mask(
                drone_scaled_full,
                ref_crop_vis * 255,
                color=(0, 255, 0),
                alpha=args.overlay_alpha,
                dilate_iters=args.overlay_dilate,
            )
            _save_image(
                out_dir / f"drone_scaled_full_overlay_{ref_label}.png",
                drone_scaled_full_overlay_ref,
            )

    match_iou = None
    match_precision = None
    match_recall = None
    match_f1 = None
    match_forward_mean = None
    match_reverse_mean = None
    match_forward_p90 = None
    match_reverse_p90 = None
    match_chamfer_mean = None
    match_chamfer_p90 = None
    match_keypoint_inlier = None
    match_keypoint_median = None
    mask_h_full, mask_w_full = best_mask.shape[:2]
    ref_x0 = max(0, off_x)
    ref_y0 = max(0, off_y)
    ref_x1 = min(best_match_map.shape[1], off_x + mask_w_full)
    ref_y1 = min(best_match_map.shape[0], off_y + mask_h_full)
    crop_w_full = ref_x1 - ref_x0
    crop_h_full = ref_y1 - ref_y0
    if crop_w_full > 0 and crop_h_full > 0:
        mask_x0 = ref_x0 - off_x
        mask_y0 = ref_y0 - off_y
        pred_crop = best_mask[
            mask_y0 : mask_y0 + crop_h_full,
            mask_x0 : mask_x0 + crop_w_full,
        ] > 0
        ref_crop = best_match_map[
            ref_y0 : ref_y0 + crop_h_full,
            ref_x0 : ref_x0 + crop_w_full,
        ] > 0
        inter = int(np.logical_and(pred_crop, ref_crop).sum())
        union = int(np.logical_or(pred_crop, ref_crop).sum())
        pred_sum = int(pred_crop.sum())
        ref_sum = int(ref_crop.sum())
        match_iou = inter / union if union > 0 else 0.0
        tol_px = max(0, int(args.match_tolerance_px))
        if tol_px > 0:
            ref_dil = _dilate_mask(ref_crop.astype(np.uint8), tol_px) > 0
            pred_dil = _dilate_mask(pred_crop.astype(np.uint8), tol_px) > 0
        else:
            ref_dil = ref_crop
            pred_dil = pred_crop
        overlap_pred = int(np.logical_and(pred_crop, ref_dil).sum())
        overlap_ref = int(np.logical_and(ref_crop, pred_dil).sum())
        match_precision = overlap_pred / pred_sum if pred_sum > 0 else None
        match_recall = overlap_ref / ref_sum if ref_sum > 0 else None
        if match_precision is not None and match_recall is not None:
            denom = match_precision + match_recall
            if denom > 0:
                match_f1 = 2.0 * match_precision * match_recall / denom
        if pred_sum > 0 and ref_sum > 0:
            dist_window = _distance_transform(ref_crop.astype(np.uint8))
            dist_rev = _distance_transform(pred_crop.astype(np.uint8))
            forward_vals = dist_window[pred_crop]
            reverse_vals = dist_rev[ref_crop]
            if forward_vals.size:
                match_forward_mean = float(forward_vals.mean())
                match_forward_p90 = float(np.percentile(forward_vals, 90))
            if reverse_vals.size:
                match_reverse_mean = float(reverse_vals.mean())
                match_reverse_p90 = float(np.percentile(reverse_vals, 90))
            if match_forward_mean is not None and match_reverse_mean is not None:
                match_chamfer_mean = 0.5 * (match_forward_mean + match_reverse_mean)
            if match_forward_p90 is not None and match_reverse_p90 is not None:
                match_chamfer_p90 = max(match_forward_p90, match_reverse_p90)
            drone_skel = _skeletonize_mask(pred_crop.astype(np.uint8))
            map_skel = _skeletonize_mask(ref_crop.astype(np.uint8))
            drone_kp = _extract_keypoints(drone_skel)
            map_kp = _extract_keypoints(map_skel)
            inlier_j, med_j = _match_keypoints(
                drone_kp["junctions"], map_kp["junctions"], float(args.match_keypoint_tol_px)
            )
            inlier_e, med_e = _match_keypoints(
                drone_kp["endpoints"], map_kp["endpoints"], float(args.match_keypoint_tol_px)
            )
            inliers = [val for val in (inlier_j, inlier_e) if val is not None]
            medians = [val for val in (med_j, med_e) if val is not None]
            if inliers:
                match_keypoint_inlier = float(np.mean(inliers))
            if medians:
                match_keypoint_median = float(np.mean(medians))

    effective_mpp = map_gsd_mpp * float(best_scale_to_full)
    match_chamfer_mean_m = None
    match_chamfer_p90_m = None
    match_keypoint_median_m = None
    if match_chamfer_mean is not None:
        match_chamfer_mean_m = match_chamfer_mean * effective_mpp
    if match_chamfer_p90 is not None:
        match_chamfer_p90_m = match_chamfer_p90 * effective_mpp
    if match_keypoint_median is not None:
        match_keypoint_median_m = match_keypoint_median * effective_mpp

    match_ok = True
    if match_f1 is None:
        match_ok = False
    if match_precision is None or match_recall is None:
        match_ok = False
    if args.match_min_f1 > 0 and match_f1 is not None and match_f1 < args.match_min_f1:
        match_ok = False
    if args.match_min_precision > 0 and match_precision is not None and match_precision < args.match_min_precision:
        match_ok = False
    if args.match_min_recall > 0 and match_recall is not None and match_recall < args.match_min_recall:
        match_ok = False
    if args.match_max_mean_dist_m > 0 and match_chamfer_mean_m is not None:
        if match_chamfer_mean_m > args.match_max_mean_dist_m:
            match_ok = False
    if args.match_max_p90_dist_m > 0 and match_chamfer_p90_m is not None:
        if match_chamfer_p90_m > args.match_max_p90_dist_m:
            match_ok = False
    if (
        args.match_min_keypoint_inlier > 0
        and match_keypoint_inlier is not None
        and match_keypoint_inlier < args.match_min_keypoint_inlier
    ):
        match_ok = False

    if args.reject_unless_ok and not match_ok:
        lat = None
        lon = None

    gt = _load_ground_truth(drone_path)
    error_km = None
    if gt is not None and lat is not None and lon is not None:
        error_km = _haversine_km(lat, lon, gt[0], gt[1])

    report = (
        f"drone_image={drone_path}\n"
        f"satellite={sat_path}\n"
        f"backend={backend_name}\n"
        f"backend_config={json.dumps(backend_effective, sort_keys=True)}\n"
        f"score={best['score']:.4f}\n"
        f"score_combined={best.get('score_combined')}\n"
        f"scale={best['scale']}\n"
        f"rotation={best['rotation']}\n"
        f"offset={best['offset']}\n"
        f"offset_full=({full_off_x}, {full_off_y})\n"
        f"search_origin=({search_origin_col}, {search_origin_row})\n"
        f"search_center={args.search_center}\n"
        f"search_center_used={search_center_used}\n"
        f"search_lat={search_lat}\n"
        f"search_lon={search_lon}\n"
        f"search_offset_north_m={args.search_offset_north_m}\n"
        f"search_offset_east_m={args.search_offset_east_m}\n"
        f"search_radius_m={args.search_radius_m if search_window else None}\n"
        f"search_window={search_window}\n"
        f"osm_cache_only={args.osm_cache_only}\n"
        f"osm_timeout={args.osm_timeout}\n"
        f"osm_rate_limit={args.osm_rate_limit}\n"
        f"sat_crs={full_crs}\n"
        f"sat_origin=({sat_origin_col}, {sat_origin_row})\n"
        f"match_stride={match_stride}\n"
        f"match_max_size={args.match_max_size}\n"
        f"best_origin=({best_origin_col}, {best_origin_row})\n"
        f"best_scale_to_full={best_scale_to_full}\n"
        f"scale_to_full_x={scale_to_full_x}\n"
        f"scale_to_full_y={scale_to_full_y}\n"
        f"sat_max_size={args.sat_max_size}\n"
        f"height_m={drone_meta.get('height')}\n"
        f"fov_deg={args.fov_deg}\n"
        f"map_gsd_mpp={map_gsd_mpp}\n"
        f"map_gsd_x={map_gsd_x}\n"
        f"map_gsd_y={map_gsd_y}\n"
        f"drone_gsd_mpp={drone_gsd_mpp}\n"
        f"scale_est={scale_est}\n"
        f"rotation_drone_angle={rot_drone_angle}\n"
        f"rotation_map_angle={rot_map_angle}\n"
        f"rotation_base={rot_base}\n"
        f"rotation_range={args.rotation_range}\n"
        f"rotation_step={args.rotation_step}\n"
        f"windowed_search={args.windowed_search}\n"
        f"window_pad_ratio={args.window_pad_ratio}\n"
        f"window_stride_ratio={args.window_stride_ratio}\n"
        f"refine_pad={args.refine_pad}\n"
        f"refine_scale_range={args.refine_scale_range}\n"
        f"refine_scale_steps={args.refine_scale_steps}\n"
        f"refine_rotation_range={args.refine_rotation_range}\n"
        f"refine_rotation_step={args.refine_rotation_step}\n"
        f"template_road_px={best['road_px']}\n"
        f"match_iou={match_iou}\n"
        f"match_precision={match_precision}\n"
        f"match_recall={match_recall}\n"
        f"match_f1={match_f1}\n"
        f"match_forward_mean_px={match_forward_mean}\n"
        f"match_reverse_mean_px={match_reverse_mean}\n"
        f"match_forward_p90_px={match_forward_p90}\n"
        f"match_reverse_p90_px={match_reverse_p90}\n"
        f"match_chamfer_mean_px={match_chamfer_mean}\n"
        f"match_chamfer_p90_px={match_chamfer_p90}\n"
        f"match_chamfer_mean_m={match_chamfer_mean_m}\n"
        f"match_chamfer_p90_m={match_chamfer_p90_m}\n"
        f"match_keypoint_inlier={match_keypoint_inlier}\n"
        f"match_keypoint_median_px={match_keypoint_median}\n"
        f"match_keypoint_median_m={match_keypoint_median_m}\n"
        f"match_ok={match_ok}\n"
        f"lat={lat}\n"
        f"lon={lon}\n"
    )
    if gt is not None:
        report += f"gt_lat={gt[0]}\n"
        report += f"gt_lon={gt[1]}\n"
        report += f"error_km={error_km}\n"
    (out_dir / "result.txt").write_text(report, encoding="utf-8")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_header = [
            "image",
            "satellite",
            "score",
            "score_combined",
            "scale",
            "rotation",
            "offset_x",
            "offset_y",
            "offset_full_x",
            "offset_full_y",
            "search_center",
            "search_center_used",
            "search_lat",
            "search_lon",
            "search_offset_north_m",
            "search_offset_east_m",
            "match_stride",
            "match_max_size",
            "best_scale_to_full",
            "scale_to_full_x",
            "scale_to_full_y",
            "lat",
            "lon",
            "gt_lat",
            "gt_lon",
            "error_km",
            "match_iou",
            "match_precision",
            "match_recall",
            "match_f1",
            "match_chamfer_mean_px",
            "match_chamfer_p90_px",
            "match_chamfer_mean_m",
            "match_chamfer_p90_m",
            "match_keypoint_inlier",
            "match_keypoint_median_px",
            "match_keypoint_median_m",
            "match_ok",
            "template_road_px",
            "scale_est",
            "windowed_search",
            "window_pad_ratio",
            "window_stride_ratio",
            "osm_cache_only",
            "osm_timeout",
            "osm_rate_limit",
        ]
        is_new = not summary_path.exists()
        active_header = summary_header
        if not is_new:
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if first_line:
                    active_header = [col.strip() for col in first_line.split(",") if col.strip()]
            except Exception:
                active_header = summary_header
        with summary_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(active_header)
            row_map = {
                "image": drone_path.name,
                "satellite": sat_path.name,
                "score": best["score"],
                "score_combined": best.get("score_combined"),
                "scale": best["scale"],
                "rotation": best["rotation"],
                "offset_x": best["offset"][0],
                "offset_y": best["offset"][1],
                "offset_full_x": full_off_x,
                "offset_full_y": full_off_y,
                "search_center": args.search_center,
                "search_center_used": search_center_used,
                "search_lat": search_lat,
                "search_lon": search_lon,
                "search_offset_north_m": args.search_offset_north_m,
                "search_offset_east_m": args.search_offset_east_m,
                "match_stride": match_stride,
                "match_max_size": args.match_max_size,
                "best_scale_to_full": best_scale_to_full,
                "scale_to_full_x": scale_to_full_x,
                "scale_to_full_y": scale_to_full_y,
                "lat": lat,
                "lon": lon,
                "gt_lat": gt[0] if gt else None,
                "gt_lon": gt[1] if gt else None,
                "error_km": error_km,
                "match_iou": match_iou,
                "match_precision": match_precision,
                "match_recall": match_recall,
                "match_f1": match_f1,
                "match_chamfer_mean_px": match_chamfer_mean,
                "match_chamfer_p90_px": match_chamfer_p90,
                "match_chamfer_mean_m": match_chamfer_mean_m,
                "match_chamfer_p90_m": match_chamfer_p90_m,
                "match_keypoint_inlier": match_keypoint_inlier,
                "match_keypoint_median_px": match_keypoint_median,
                "match_keypoint_median_m": match_keypoint_median_m,
                "match_ok": match_ok,
                "template_road_px": best["road_px"],
                "scale_est": scale_est,
                "windowed_search": args.windowed_search,
                "window_pad_ratio": args.window_pad_ratio,
                "window_stride_ratio": args.window_stride_ratio,
                "osm_cache_only": args.osm_cache_only,
                "osm_timeout": args.osm_timeout,
                "osm_rate_limit": args.osm_rate_limit,
            }
            writer.writerow([row_map.get(col) for col in active_header])

    if args.qgis:
        scale_full_x = float(best["scale"]) * float(scale_to_full_x)
        scale_full_y = float(best["scale"]) * float(scale_to_full_y)
        _export_qgis_outputs(
            out_dir=out_dir,
            drone_rgb=drone_rgb,
            drone_mask=drone_mask,
            sat_path=sat_path,
            full_transform=full_transform,
            full_crs=full_crs,
            full_w=full_w,
            full_h=full_h,
            center_row=float(orig_row),
            center_col=float(orig_col),
            scale_full_x=scale_full_x,
            scale_full_y=scale_full_y,
            rotation_deg=angle,
            lat=lat,
            lon=lon,
            use_osm=args.use_osm,
            osm_cache_dir=Path(args.osm_cache),
            osm_network_type=args.osm_network_type,
            osm_dilate_px=args.osm_dilate_px,
            osm_cache_only=args.osm_cache_only,
            osm_timeout=args.osm_timeout,
            osm_rate_limit=args.osm_rate_limit,
            full_osm_mask=full_mask,
            pad_px=args.qgis_pad_px,
        )
    return 0


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray | None:
    if scale <= 0:
        return None
    h, w = mask.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w < 2 or new_h < 2:
        return None
    try:
        import cv2  # type: ignore

        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(mask).resize((new_w, new_h), resample=Image.NEAREST))


if __name__ == "__main__":
    raise SystemExit(main())
