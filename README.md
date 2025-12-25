# GPS-denied localization, v2: road-mask matching (Dec 2025)

Second attempt at locating a drone from a downward camera image and a satellite map, on the
[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) dataset. Archived here for history.
The current version is **[gps-denied-localization](https://github.com/hakan4vg/gps-denied-localization)**.

## Pipeline

1. Extract a road mask from the drone image with SAM-Road (TopoNet, ViT-B), or a classic
   Canny-based extractor (`road_extractors/`).
2. Build a road mask for the map area from OpenStreetMap (`scripts/prebuild_osm.py`) or from the
   satellite image.
3. Chamfer template matching over scale × rotation × position (`scripts/localize_roads.py`),
   then rescore the top candidates by road-overlap precision/recall/F1.
4. Export results as GeoTIFF/GeoJSON/QGIS project for inspection.

## Why it was replaced

- **Slow on CPU.** SAM-Road runs ~256 ViT-B passes per image (16×16 patches of 512 px):
  minutes per image.
- **Brute-force search.** 7 scales × 13 rotations × full-map template matching, and that only
  covered ±30° around a metadata angle.
- **Depended on priors.** The default `--search-center metadata` searches a 2 km radius around
  the ground-truth position from the dataset CSV and takes scale from the recorded height. This
  was a deliberate shortcut for a GIS class demo. Without it, the sensor-free search did not work.
- **Mismatched road sources.** SAM-Road masks (roads with width) vs OSM centerlines (no width,
  missing tracks) produced false and missed matches.

The restart replaces the search with rotation- and scale-invariant geometric hashing of
landmark points, and extracts points the same way on both the drone and satellite side.

Model checkpoints, the SAM-Road code (`third_party/`) and the dataset are not included.
