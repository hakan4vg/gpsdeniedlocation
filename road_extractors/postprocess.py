from __future__ import annotations

import numpy as np


def _maybe_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def postprocess_mask(
    mask: np.ndarray,
    threshold: float | None = 0.5,
    close_iters: int = 0,
    open_iters: int = 0,
    kernel_size: int = 3,
    min_component_area: int = 0,
    skeletonize: bool = False,
    prune_iters: int = 0,
) -> np.ndarray:
    if mask.dtype != np.uint8:
        if threshold is None:
            threshold = 0.5
        mask = (mask >= threshold).astype(np.uint8) * 255
    else:
        if threshold is not None:
            thresh = int(round(255 * threshold))
            mask = (mask >= thresh).astype(np.uint8) * 255

    if close_iters <= 0 and open_iters <= 0 and min_component_area <= 0 and not skeletonize and prune_iters <= 0:
        return mask

    cv2 = _maybe_import_cv2()
    if cv2 is None:
        return mask

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    if open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if min_component_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            keep = np.zeros_like(mask)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_component_area:
                    keep[labels == label] = 255
            mask = keep
    if prune_iters > 0 and not skeletonize:
        skeletonize = True
    if skeletonize:
        mask = _skeletonize(mask)
    if prune_iters > 0:
        mask = _prune_skeleton(mask, prune_iters)
    return mask


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    cv2 = _maybe_import_cv2()
    if cv2 is None:
        return mask
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(mask)
    img = mask.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _prune_skeleton(mask: np.ndarray, prune_iters: int) -> np.ndarray:
    cv2 = _maybe_import_cv2()
    if cv2 is None:
        return mask
    pruned = mask.copy()
    kernel = np.ones((3, 3), dtype=np.float32)
    for _ in range(prune_iters):
        binary = (pruned > 0).astype(np.float32)
        if binary.sum() == 0:
            break
        neighbors = cv2.filter2D(binary, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        neighbor_count = neighbors - binary
        endpoints = (binary > 0.5) & (neighbor_count <= 1.0)
        if not endpoints.any():
            break
        pruned[endpoints] = 0
    return pruned
