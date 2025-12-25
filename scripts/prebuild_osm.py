from __future__ import annotations

import argparse
import os
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _bounds_to_wgs84(
    crs: Any, left: float, bottom: float, right: float, top: float
) -> tuple[float, float, float, float]:
    xs = [left, right, left, right]
    ys = [bottom, bottom, top, top]
    lons, lats = _transform_coords(crs, "EPSG:4326", xs, ys)
    north = max(lats)
    south = min(lats)
    east = max(lons)
    west = min(lons)
    return north, south, east, west


def _load_csv_ranges(csv_path: Path) -> dict[str, tuple[float, float, float, float]]:
    if not csv_path.exists():
        return {}
    ranges: dict[str, tuple[float, float, float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("mapname")
            if not name:
                continue
            try:
                north = float(row["LT_lat_map"])
                west = float(row["LT_lon_map"])
                south = float(row["RB_lat_map"])
                east = float(row["RB_lon_map"])
            except Exception:
                continue
            ranges[name] = (north, south, east, west)
    return ranges


def _rasterize_edges(
    edges: Any,
    out_shape: tuple[int, int],
    transform: Any,
    dst_crs: Any | None,
    osm_dilate_px: int,
    cache_path: Path,
) -> Path:
    import rasterio.features

    if edges is None or getattr(edges, "empty", False):
        shapes = []
    else:
        edges = edges[edges.geometry.notnull()]
        if (
            dst_crs is not None
            and getattr(edges, "crs", None) is not None
            and edges.crs != dst_crs
        ):
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
    return cache_path


def _build_osm_mask(
    out_shape: tuple[int, int],
    transform: Any,
    dst_crs: Any | None,
    cache_dir: Path,
    cache_tag: str,
    network_type: str,
    osm_dilate_px: int,
    bbox: tuple[float, float, float, float],
    osm_timeout: int,
    osm_rate_limit: bool,
) -> Path:
    import osmnx as ox

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{cache_tag}_osm_v2_{out_shape[1]}x{out_shape[0]}_{network_type}_d{osm_dilate_px}.npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return cache_path

    north, south, east, west = bbox
    ox.settings.timeout = int(osm_timeout)
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
        edges = ox.graph_to_gdfs(
            graph, nodes=False, edges=True, fill_edge_geometry=True
        )
    except Exception:
        tags = {"highway": True}
        try:
            edges = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
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

    return _rasterize_edges(
        edges,
        out_shape=out_shape,
        transform=transform,
        dst_crs=dst_crs,
        osm_dilate_px=osm_dilate_px,
        cache_path=cache_path,
    )


def _build_osm_mask_from_pbf(
    pbf_path: Path,
    out_shape: tuple[int, int],
    transform: Any,
    dst_crs: Any | None,
    cache_dir: Path,
    cache_tag: str,
    network_type: str,
    osm_dilate_px: int,
    bbox: tuple[float, float, float, float],
) -> Path:
    try:
        from pyrosm import OSM  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "pyrosm is required for --pbf mode. Install in the venv: pip install pyrosm"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{cache_tag}_osm_v2_{out_shape[1]}x{out_shape[0]}_{network_type}_d{osm_dilate_px}.npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return cache_path

    north, south, east, west = bbox
    bounding_box = [west, south, east, north]
    osm = OSM(str(pbf_path), bounding_box=bounding_box)
    try:
        edges = osm.get_network(network_type=network_type)
    except Exception:
        edges = osm.get_network(network_type="driving")

    return _rasterize_edges(
        edges,
        out_shape=out_shape,
        transform=transform,
        dst_crs=dst_crs,
        osm_dilate_px=osm_dilate_px,
        cache_path=cache_path,
    )


def _collect_satellites(dataset_root: Path) -> list[Path]:
    patterns = [
        "**/satellite*.tif",
        "**/satellite*.tiff",
        "**/SATELLITE*.tif",
        "**/SATELLITE*.tiff",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(dataset_root.glob(pattern))
    return sorted({p.resolve() for p in found})


def _gdal_available() -> bool:
    import shutil

    return (
        shutil.which("ogr2ogr") is not None
        and shutil.which("gdal_rasterize") is not None
    )


def _highway_filter(network_type: str) -> str:
    if network_type in {"drive", "driving", "drive_service"}:
        values = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "service",
            "living_street",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        ]
        values_str = ",".join(f"'{val}'" for val in values)
        return f"highway IN ({values_str})"
    return "highway IS NOT NULL"


def _build_osm_mask_from_pbf_gdal(
    pbf_path: Path,
    out_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    dst_crs: Any | None,
    cache_dir: Path,
    cache_tag: str,
    network_type: str,
    osm_dilate_px: int,
    bbox: tuple[float, float, float, float],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{cache_tag}_osm_v2_{out_shape[1]}x{out_shape[0]}_{network_type}_d{osm_dilate_px}.npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return cache_path

    north, south, east, west = bbox
    left, bottom, right, top = bounds
    where = _highway_filter(network_type)
    if not _gdal_available():
        raise SystemExit("ogr2ogr and gdal_rasterize are required for --pbf-mode gdal")

    with tempfile.TemporaryDirectory(prefix="osm_extract_") as tmp_dir:
        vec_path = Path(tmp_dir) / "roads.gpkg"
        rast_path = Path(tmp_dir) / "roads.tif"
        osmconf_path = Path(tmp_dir) / "osmconf.ini"
        osmconf_path.write_text(
            "\n".join(
                [
                    "[osmconf]",
                    "closed_ways_are_polygons=no",
                    "attribute_name_laundering=yes",
                    "report_all_nodes=no",
                    "",
                    "[points]",
                    "other_tags=no",
                    "attributes=",
                    "",
                    "[lines]",
                    "other_tags=no",
                    "attributes=highway",
                    "",
                    "[multilinestrings]",
                    "other_tags=no",
                    "attributes=highway",
                    "",
                    "[multipolygons]",
                    "other_tags=no",
                    "attributes=",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        ogr_cmd = [
            "ogr2ogr",
            "-f",
            "GPKG",
            str(vec_path),
            str(pbf_path),
            "lines",
            "-oo",
            f"CONFIG_FILE={osmconf_path}",
            "-spat",
            str(west),
            str(south),
            str(east),
            str(north),
            "-select",
            "highway",
            "-where",
            where,
            "-overwrite",
            "-skipfailures",
        ]
        if dst_crs is not None:
            ogr_cmd.extend(["-t_srs", str(dst_crs)])
        subprocess.run(ogr_cmd, check=True)

        width, height = out_shape[1], out_shape[0]
        rast_cmd = [
            "gdal_rasterize",
            "-burn",
            "1",
            "-l",
            "lines",
            "-ot",
            "Byte",
            "-ts",
            str(width),
            str(height),
            "-te",
            str(left),
            str(bottom),
            str(right),
            str(top),
            str(vec_path),
            str(rast_path),
        ]
        subprocess.run(rast_cmd, check=True)

        try:
            import rasterio
        except Exception as exc:
            raise SystemExit("rasterio is required to read raster output") from exc
        with rasterio.open(rast_path) as src:
            data = src.read(1, out_dtype=np.uint8)

    if osm_dilate_px > 0:
        try:
            import cv2  # type: ignore

            kernel = np.ones((3, 3), dtype=np.uint8)
            data = cv2.dilate(data, kernel, iterations=osm_dilate_px)
        except Exception:
            pass

    np.save(cache_path, data.astype(np.uint8))
    return cache_path


def _build_osm_mask_from_pbf_gdal_direct(
    pbf_path: Path,
    out_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    dst_crs: Any | None,
    cache_dir: Path,
    cache_tag: str,
    network_type: str,
    osm_dilate_px: int,
    bbox: tuple[float, float, float, float],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{cache_tag}_osm_v2_{out_shape[1]}x{out_shape[0]}_{network_type}_d{osm_dilate_px}.npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        return cache_path
    if dst_crs is not None and str(dst_crs) not in {"EPSG:4326", "WGS84"}:
        raise SystemExit("pbf-mode gdal-direct requires EPSG:4326 GeoTIFFs")

    left, bottom, right, top = bounds
    where = _highway_filter(network_type)
    if not _gdal_available():
        raise SystemExit("gdal_rasterize is required for --pbf-mode gdal-direct")

    with tempfile.TemporaryDirectory(prefix="osm_rast_") as tmp_dir:
        osmconf_path = Path(tmp_dir) / "osmconf.ini"
        osmconf_path.write_text(
            "\n".join(
                [
                    "[osmconf]",
                    "closed_ways_are_polygons=no",
                    "attribute_name_laundering=yes",
                    "report_all_nodes=no",
                    "",
                    "[points]",
                    "other_tags=no",
                    "attributes=",
                    "",
                    "[lines]",
                    "other_tags=no",
                    "attributes=highway",
                    "",
                    "[multilinestrings]",
                    "other_tags=no",
                    "attributes=highway",
                    "",
                    "[multipolygons]",
                    "other_tags=no",
                    "attributes=",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        rast_path = Path(tmp_dir) / "roads.tif"
        width, height = out_shape[1], out_shape[0]
        rast_cmd = [
            "gdal_rasterize",
            "-oo",
            "INTERLEAVED_READING=YES",
            "-oo",
            f"CONFIG_FILE={osmconf_path}",
            "-burn",
            "1",
            "-at",
            "-l",
            "lines",
            "-ot",
            "Byte",
            "-ts",
            str(width),
            str(height),
            "-te",
            str(left),
            str(bottom),
            str(right),
            str(top),
            "-where",
            where,
            str(pbf_path),
            str(rast_path),
        ]
        env = os.environ.copy()
        env["OGR_INTERLEAVED_READING"] = "YES"
        subprocess.run(rast_cmd, check=True, env=env)

        try:
            import rasterio
        except Exception as exc:
            raise SystemExit("rasterio is required to read raster output") from exc
        with rasterio.open(rast_path) as src:
            data = src.read(1, out_dtype=np.uint8)

    if osm_dilate_px > 0:
        try:
            import cv2  # type: ignore

            kernel = np.ones((3, 3), dtype=np.uint8)
            data = cv2.dilate(data, kernel, iterations=osm_dilate_px)
        except Exception:
            pass

    np.save(cache_path, data.astype(np.uint8))
    return cache_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prebuild OSM road masks for satellite GeoTIFFs"
    )
    parser.add_argument("--dataset-root", default="UAV_VisLoc_dataset")
    parser.add_argument("--osm-cache", default="cache/osm_masks")
    parser.add_argument("--osm-network-type", default="drive")
    parser.add_argument("--osm-dilate-px", type=int, default=0)
    parser.add_argument("--osm-timeout", type=int, default=480)
    parser.add_argument("--osm-rate-limit", action="store_true", default=False)
    parser.add_argument(
        "--pbf",
        type=str,
        default=None,
        help="Path to local .osm.pbf for offline builds",
    )
    parser.add_argument(
        "--pbf-mode",
        default="auto",
        choices=["auto", "pyrosm", "gdal", "gdal-direct"],
    )
    parser.add_argument("--use-csv-range", action="store_true")
    parser.add_argument(
        "--csv-path",
        default="UAV_VisLoc_dataset/satellite_ coordinates_range.csv",
    )
    parser.add_argument(
        "--maps", default=None, help="Comma-separated satellite basenames to include"
    )
    parser.add_argument("--max-maps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_dir = Path(args.osm_cache)
    include = None
    if args.maps:
        include = {name.strip() for name in args.maps.split(",") if name.strip()}

    csv_ranges = _load_csv_ranges(Path(args.csv_path)) if args.use_csv_range else {}

    satellites = _collect_satellites(dataset_root)
    if include is not None:
        satellites = [
            path for path in satellites if path.stem in include or path.name in include
        ]
    if args.max_maps is not None:
        satellites = satellites[: max(0, int(args.max_maps))]
    if not satellites:
        raise SystemExit("No satellite GeoTIFFs found")

    pbf_path = Path(args.pbf) if args.pbf else None
    use_gdal = False
    use_gdal_direct = False
    if pbf_path is not None:
        if args.pbf_mode == "gdal":
            if not _gdal_available():
                raise SystemExit(
                    "pbf-mode gdal requested, but ogr2ogr/gdal_rasterize not found"
                )
            use_gdal = True
        elif args.pbf_mode == "gdal-direct":
            if not _gdal_available():
                raise SystemExit(
                    "pbf-mode gdal-direct requested, but gdal_rasterize not found"
                )
            use_gdal_direct = True
        elif args.pbf_mode == "auto":
            use_gdal = _gdal_available()
    for sat_path in satellites:
        print(f"Processing {sat_path}...")
        try:
            import rasterio
        except Exception as exc:
            raise SystemExit("rasterio is required") from exc

        with rasterio.open(sat_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        if args.use_csv_range and sat_path.name in csv_ranges:
            bbox = csv_ranges[sat_path.name]
        else:
            bbox = _bounds_to_wgs84(
                crs, bounds.left, bounds.bottom, bounds.right, bounds.top
            )

        cache_name = f"{sat_path.stem}_osm_v2_{width}x{height}_{args.osm_network_type}_d{args.osm_dilate_px}.npy"
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            print(f"  cache hit: {cache_path}")
            continue

        if args.dry_run:
            print(f"  dry-run: would fetch bbox={bbox} -> {cache_path}")
            continue

        if pbf_path is not None and use_gdal_direct:
            out_path = _build_osm_mask_from_pbf_gdal_direct(
                pbf_path=pbf_path,
                out_shape=(height, width),
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                dst_crs=crs,
                cache_dir=cache_dir,
                cache_tag=sat_path.stem,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=bbox,
            )
        elif pbf_path is not None and use_gdal:
            out_path = _build_osm_mask_from_pbf_gdal(
                pbf_path=pbf_path,
                out_shape=(height, width),
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                dst_crs=crs,
                cache_dir=cache_dir,
                cache_tag=sat_path.stem,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=bbox,
            )
        elif pbf_path is not None:
            out_path = _build_osm_mask_from_pbf(
                pbf_path=pbf_path,
                out_shape=(height, width),
                transform=transform,
                dst_crs=crs,
                cache_dir=cache_dir,
                cache_tag=sat_path.stem,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=bbox,
            )
        else:
            out_path = _build_osm_mask(
                out_shape=(height, width),
                transform=transform,
                dst_crs=crs,
                cache_dir=cache_dir,
                cache_tag=sat_path.stem,
                network_type=args.osm_network_type,
                osm_dilate_px=args.osm_dilate_px,
                bbox=bbox,
                osm_timeout=args.osm_timeout,
                osm_rate_limit=args.osm_rate_limit,
            )
        print(f"  wrote {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
