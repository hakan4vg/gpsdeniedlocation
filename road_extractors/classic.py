from __future__ import annotations

import numpy as np

from .base import RoadExtractor
from .postprocess import postprocess_mask


def _maybe_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for the classic backend") from exc
    return cv2


class ClassicRoadExtractor(RoadExtractor):
    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        close_iters: int = 1,
        open_iters: int = 0,
        kernel_size: int = 3,
    ) -> None:
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.close_iters = close_iters
        self.open_iters = open_iters
        self.kernel_size = kernel_size

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        cv2 = _maybe_import_cv2()
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected an HxWxC image with 3 channels")
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        if self.blur_kernel > 0:
            k = self.blur_kernel
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        mask = postprocess_mask(
            edges,
            threshold=None,
            close_iters=self.close_iters,
            open_iters=self.open_iters,
            kernel_size=self.kernel_size,
        )
        return mask
