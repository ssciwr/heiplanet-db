#!/usr/bin/env python3
"""
A script I generated with AI to downsample the world country borders reference image we were provided to a suitable geojson size for frontend use.

I think as data updates do come to the other types of data (e.g. NUTS) which exist and are defined here on the backend, this makes sense to live here,
but the exported image then gets used and saved on the frontent.

Downsample GeoJSON-like boundary files to a target max size.
Usage example:
python _scripts/downsample_geojson.py "World Bank Official Boundaries - Admin 0(1).png" --output "/
  downsample.geojson" --max-mb 6


Example output:
step=1 size=80140386
step=2 size=41256344
step=3 size=28353567
step=4 size=21702896
step=5 size=17520696
step=6 size=14747263
step=8 size=11331385
step=10 size=9310428
step=12 size=7977292
step=15 size=6655828
step=20 size=5353581

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_point(item: object) -> bool:
    return (
        isinstance(item, list)
        and len(item) >= 2
        and isinstance(item[0], (int, float))
        and isinstance(item[1], (int, float))
    )


def simplify_line(line: list, step: int) -> list:
    if not line:
        return line
    points = line[::step] if step > 1 else line[:]
    if points[0] != line[0]:
        points.insert(0, line[0])
    if points[-1] != line[-1]:
        points.append(line[-1])
    return [[round(p[0], 5), round(p[1], 5)] + p[2:] for p in points]


def simplify_coords(coords: object, step: int) -> object:
    if not isinstance(coords, list):
        return coords
    if coords and is_point(coords[0]):
        return simplify_line(coords, step)
    return [simplify_coords(item, step) for item in coords]


def simplify_geom(geom: dict, step: int) -> dict:
    out = dict(geom)
    if "coordinates" in out:
        out["coordinates"] = simplify_coords(out["coordinates"], step)
    if "geometries" in out and isinstance(out["geometries"], list):
        out["geometries"] = [simplify_geom(item, step) for item in out["geometries"]]
    return out


def downsample_geojson(data: dict, step: int) -> dict:
    out = dict(data)
    features = out.get("features")
    if isinstance(features, list):
        next_features = []
        for feature in features:
            updated = dict(feature)
            if isinstance(feature.get("geometry"), dict):
                updated["geometry"] = simplify_geom(feature["geometry"], step)
            next_features.append(updated)
        out["features"] = next_features
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample boundary GeoJSON to target size.")
    parser.add_argument("input", type=Path, help="Input file path (GeoJSON text)")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--max-mb", type=float, default=6.0, help="Target max size in MB")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem} - downsampled.geojson")
    max_bytes = int(args.max_mb * 1024 * 1024)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
    for step in steps:
        output_data = downsample_geojson(data, step)
        text = json.dumps(output_data, separators=(",", ":"))

        output_path.write_text(text, encoding="utf-8")
        size = output_path.stat().st_size
        print(f"step={step} size={size}")
        if size < max_bytes:
            print(f"output={output_path}")
            print(f"final_size_bytes={size}")
            return

    raise SystemExit(f"Could not reduce file below {args.max_mb} MB")


if __name__ == "__main__":
    main()
