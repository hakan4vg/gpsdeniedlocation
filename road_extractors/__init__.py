from __future__ import annotations

from typing import Any

from .classic import ClassicRoadExtractor
from .sam_road import SamRoadExtractor


def get_extractor(name: str, **kwargs: Any):
    key = name.lower().strip()
    if key in {"sam_road", "sam-road", "samroad"}:
        return SamRoadExtractor(**kwargs)
    if key in {"classic", "canny"}:
        return ClassicRoadExtractor(**kwargs)
    raise ValueError(f"Unknown road extractor backend: {name}")
