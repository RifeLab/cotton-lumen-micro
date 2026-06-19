#!/usr/bin/env python
"""Shared contour measurement utilities for cotton fiber analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import acos, degrees, sqrt
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None


Point = tuple[float, float]
Axis = tuple[Point, Point]

CSV_HEADER = [
    "index",
    "filename",
    "class",
    "area",
    "boundingBoxArea",
    "minRectArea",
    "hullArea",
    "perimeter",
    "isConvex",
    "circularity",
    "minAxisLength",
    "maxAxisLength",
    "aspectRatio",
    "minRectOrientation",
    "extent",
    "solidity",
    "equivalentDiameter",
    "ellipseMajorAxisLength",
    "ellipseMinorAxisLength",
    "ellipseOrientationAngle",
    "eccentricity",
    "position_x",
    "position_y",
]


def _require_image_dependencies() -> None:
    if cv2 is None or np is None:
        raise ModuleNotFoundError("fiber_analysis requires opencv-python and numpy to process images.")


def axis_length(axis: Optional[Axis]) -> float:
    """Return Euclidean length for a two-point axis."""
    if axis is None:
        return 0.0
    return sqrt((float(axis[0][0]) - float(axis[1][0])) ** 2 + (float(axis[0][1]) - float(axis[1][1])) ** 2)


def contour_from_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert an Nx2 point sequence to an OpenCV contour."""
    _require_image_dependencies()
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def find_longest_line(contour: np.ndarray) -> Axis:
    """Find the longest point-to-point chord in a contour."""
    _require_image_dependencies()
    points = contour.reshape((-1, 2))
    best_length = 0.0
    best_axis: Axis = ((0.0, 0.0), (0.0, 0.0))

    for i, p1 in enumerate(points):
        deltas = points[i:] - p1
        lengths = np.einsum("ij,ij->i", deltas, deltas)
        if lengths.size == 0:
            continue
        local_index = int(np.argmax(lengths))
        local_length = float(lengths[local_index])
        if local_length > best_length:
            best_length = local_length
            p2 = points[i + local_index]
            best_axis = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))

    return best_axis


def measure_smart_grain(
    contour: np.ndarray,
    *,
    epsilon_ratio: float = 0.0005,
    perpendicular_tolerance_degrees: float = 1.0,
) -> tuple[Axis, Axis]:
    """Measure an approximate major axis and the longest near-perpendicular minor axis."""
    _require_image_dependencies()
    if contour is None or len(contour) < 2:
        zero_axis: Axis = ((0.0, 0.0), (0.0, 0.0))
        return zero_axis, zero_axis

    approx_curve = cv2.approxPolyDP(contour, epsilon_ratio * cv2.arcLength(contour, True), True)
    points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx_curve]

    max_axis = find_longest_line(contour)
    max_vector = np.asarray(max_axis[1]) - np.asarray(max_axis[0])
    max_length = axis_length(max_axis)
    min_axis = max_axis
    min_axis_length = 0.0

    if max_length <= 1e-8:
        return max_axis, min_axis

    lower = 90.0 - perpendicular_tolerance_degrees
    upper = 90.0 + perpendicular_tolerance_degrees

    for start in points:
        for end in points:
            candidate_vector = np.asarray(end) - np.asarray(start)
            candidate_length = float(np.linalg.norm(candidate_vector))
            if candidate_length <= 1e-8:
                continue

            cos_theta = float(np.dot(max_vector, candidate_vector) / (max_length * candidate_length))
            angle = degrees(acos(max(min(cos_theta, 1.0), -1.0)))

            if lower <= angle <= upper and candidate_length > min_axis_length:
                min_axis = (start, end)
                min_axis_length = candidate_length

    return max_axis, min_axis


def contour_mask(image_shape: tuple[int, ...], contour: np.ndarray) -> np.ndarray:
    """Create a single-channel mask for a contour."""
    _require_image_dependencies()
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask


def is_inside(inner: np.ndarray, outer: np.ndarray, *, padding: int = 1) -> bool:
    """Return True when inner contour's bounding box sits inside outer contour's box."""
    _require_image_dependencies()
    ix, iy, iw, ih = cv2.boundingRect(np.asarray(inner))
    ox, oy, ow, oh = cv2.boundingRect(np.asarray(outer))
    return (
        ox <= ix + padding
        and oy <= iy + padding
        and ox + ow >= ix + iw - padding
        and oy + oh >= iy + ih - padding
    )


def find_cotton_lumen_pairs(
    cotton_contours: Iterable[np.ndarray],
    lumen_contours: Iterable[np.ndarray],
) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
    """Pair each cotton contour with the first lumen contour contained by it."""
    lumen = list(lumen_contours)
    pairs: list[tuple[np.ndarray, Optional[np.ndarray]]] = []

    for cotton in cotton_contours:
        lumen_match = next((candidate for candidate in lumen if is_inside(candidate, cotton)), None)
        pairs.append((cotton, lumen_match))

    return pairs


def conversion_factor_from_yellow_legend(
    image: np.ndarray,
    *,
    legend_width_microns: float = 40.0,
    yellow_lower_hsv: tuple[int, int, int] = (25, 0, 0),
    yellow_upper_hsv: tuple[int, int, int] = (36, 255, 255),
) -> float:
    """Estimate microns-per-pixel from the largest yellow legend, falling back to 1."""
    _require_image_dependencies()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array(yellow_lower_hsv), np.array(yellow_upper_hsv))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0

    _, _, width, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if width <= 0:
        return 1.0
    return float(legend_width_microns) / float(width)


def _format_number(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


@dataclass
class FiberMeasurement:
    filename: str
    class_name: str
    contour: np.ndarray
    conversion_factor: float = 1.0
    area: Optional[float] = None
    bounding_rect_area: Optional[float] = None
    min_rect_area: Optional[float] = None
    hull_area: Optional[float] = None
    perimeter: Optional[float] = None
    is_convex: Optional[bool] = None
    circularity: Optional[float] = None
    min_axis: Optional[Axis] = None
    max_axis: Optional[Axis] = None
    min_axis_length: Optional[float] = None
    max_axis_length: Optional[float] = None
    aspect_ratio: Optional[float] = None
    min_rect_orientation: Optional[float] = None
    extent: Optional[float] = None
    solidity: Optional[float] = None
    equivalent_diameter: Optional[float] = None
    ellipse_major_axis: Optional[float] = None
    ellipse_minor_axis: Optional[float] = None
    ellipse_orientation_angle: Optional[float] = None
    eccentricity: Optional[float] = None
    position: Point = (-1.0, -1.0)

    @classmethod
    def analyze(
        cls,
        filename: str,
        image: np.ndarray,
        contour: np.ndarray,
        *,
        class_name: str,
        conversion_factor: Optional[float] = None,
    ) -> "FiberMeasurement":
        _require_image_dependencies()
        factor = conversion_factor
        if factor is None:
            factor = conversion_factor_from_yellow_legend(image)

        measurement = cls(filename=filename, class_name=class_name, contour=contour, conversion_factor=factor)
        measurement._analyze()
        return measurement

    def _analyze(self) -> None:
        _require_image_dependencies()
        contour = self.contour
        factor = self.conversion_factor

        self.area = cv2.contourArea(contour) * (factor**2)
        self.hull_area = None
        self.perimeter = cv2.arcLength(contour, True) * factor
        self.is_convex = cv2.isContourConvex(contour)

        x, y, width, height = cv2.boundingRect(contour)
        self.bounding_rect_area = width * height * (factor**2)
        self.aspect_ratio = float(width) / float(height) if height else None

        min_rect = cv2.minAreaRect(contour)
        _, size, self.min_rect_orientation = min_rect
        self.min_rect_area = size[0] * size[1] * (factor**2)

        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        self.position = (float(cx), float(cy))

        hull = cv2.convexHull(contour)
        self.hull_area = cv2.contourArea(hull) * (factor**2)
        self.extent = self.area / self.bounding_rect_area if self.bounding_rect_area else None
        self.solidity = self.area / self.hull_area if self.hull_area else None
        self.equivalent_diameter = sqrt(4.0 * self.area / np.pi) if self.area else None
        self.circularity = (4.0 * np.pi * self.area) / (self.perimeter**2) if self.perimeter else None

        if len(hull) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(hull)
            self.ellipse_major_axis = max(axis_a, axis_b) * factor
            self.ellipse_minor_axis = min(axis_a, axis_b) * factor
            self.ellipse_orientation_angle = angle
            self.eccentricity = (
                self.ellipse_minor_axis / self.ellipse_major_axis if self.ellipse_major_axis else None
            )

        self.max_axis, self.min_axis = measure_smart_grain(contour)
        self.max_axis_length = axis_length(self.max_axis) * factor
        self.min_axis_length = axis_length(self.min_axis) * factor

    def to_csv_row(self, index: int) -> list[object]:
        return [
            index,
            self.filename,
            self.class_name,
            _format_number(self.area),
            _format_number(self.bounding_rect_area),
            _format_number(self.min_rect_area),
            _format_number(self.hull_area),
            _format_number(self.perimeter),
            self.is_convex,
            _format_number(self.circularity),
            _format_number(self.min_axis_length),
            _format_number(self.max_axis_length),
            _format_number(self.aspect_ratio),
            _format_number(self.min_rect_orientation),
            _format_number(self.extent),
            _format_number(self.solidity),
            _format_number(self.equivalent_diameter),
            _format_number(self.ellipse_major_axis),
            _format_number(self.ellipse_minor_axis),
            _format_number(self.ellipse_orientation_angle),
            _format_number(self.eccentricity),
            str(int(self.position[0])),
            str(int(self.position[1])),
        ]


def write_measurements_csv(path: Path, rows: Iterable[tuple[int, FiberMeasurement]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        for index, measurement in rows:
            writer.writerow(measurement.to_csv_row(index))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure contours in an OpenCV-readable mask image.")
    parser.add_argument("mask", type=Path, help="Binary or grayscale mask image.")
    parser.add_argument("--image", type=Path, help="Original image used for legend-based unit conversion.")
    parser.add_argument("--output-csv", type=Path, default=Path("fiber_measurements.csv"))
    parser.add_argument("--class-name", default="cotton", help="Class label to write to the CSV.")
    parser.add_argument("--min-area", type=float, default=0.0, help="Discard contours below this pixel area.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _require_image_dependencies()
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {args.mask}")

    image = cv2.imread(str(args.image)) if args.image else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    measurements = []
    for index, contour in enumerate(contours):
        if cv2.contourArea(contour) >= args.min_area:
            measurements.append(
                (index, FiberMeasurement.analyze(str(args.mask), image, contour, class_name=args.class_name))
            )

    write_measurements_csv(args.output_csv, measurements)
    print(f"Wrote {len(measurements)} measurements to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
