from __future__ import annotations

import numpy as np


class RoadExtractor:
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
