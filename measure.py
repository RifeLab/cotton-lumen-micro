#!/usr/bin/env python
"""Shared contour measurement utilities for cotton fiber analysis."""

import argparse
import csv
import time
from math import acos, degrees, sqrt
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None

CSV_HEADER = [
    "index",
    "filename",
    "class",
    "area",
    "pixelArea",
    "boundingBoxArea",
    "minRectArea",
    "hullArea",
    "perimeter",
    "smoothedPerimeter",
    "isConvex",
    "circularity",
    "smoothedCircularity",
    "pixelCircularity",
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
    "centroid_x",
    "centroid_y",
]

# Fraction of raw perimeter used as approxPolyDP epsilon for smoothed perimeter/circularity.
DEFAULT_SMOOTH_EPSILON_RATIO = 0.01


class MeasurementTiming(object):
    """Counts and stage times for a measurement run.

    Processed contours are the ones kept and measured. Filtered contours are the
    ones dropped by area, window, seam, duplicate, or large-contour checks.
    Window time covers each sliding-window tile, including the inference call
    inside it. Analysis time is the per-object morphology measurement.
    """

    def __init__(self, class_names=("cotton", "lumen")):
        self.class_names = tuple(dict.fromkeys(("cotton", "lumen", *tuple(class_names))))
        self.processed = {name: 0 for name in self.class_names}
        self.filtered = {name: 0 for name in self.class_names}
        self.analysis_seconds = {name: 0.0 for name in self.class_names}
        self.analysis_counts = {name: 0 for name in self.class_names}
        self.images_processed = 0
        self.windows_processed = 0
        self.window_seconds = 0.0
        self.inference_seconds = 0.0
        self.image_read_seconds = 0.0
        self.scale_seconds = 0.0
        self.filter_seconds = 0.0
        self.overlay_seconds = 0.0
        self.write_seconds = 0.0
        self.confusion_seconds = 0.0

    def record_filtered(self, class_name, count=1):
        if class_name in self.filtered and count:
            self.filtered[class_name] += count

    def record_processed(self, class_name, count=1):
        if class_name in self.processed and count:
            self.processed[class_name] += count

    def record_analysis(self, class_name, seconds):
        if class_name not in self.analysis_seconds:
            return
        self.analysis_seconds[class_name] += seconds
        self.analysis_counts[class_name] += 1

    def record_window(self, seconds):
        self.windows_processed += 1
        self.window_seconds += seconds

    def snapshot(self):
        """Return a copy of the counters so one image can be reported on its own."""
        copy = MeasurementTiming(self.class_names)
        copy.processed = dict(self.processed)
        copy.filtered = dict(self.filtered)
        copy.analysis_seconds = dict(self.analysis_seconds)
        copy.analysis_counts = dict(self.analysis_counts)
        copy.images_processed = self.images_processed
        copy.windows_processed = self.windows_processed
        copy.window_seconds = self.window_seconds
        copy.inference_seconds = self.inference_seconds
        copy.image_read_seconds = self.image_read_seconds
        copy.scale_seconds = self.scale_seconds
        copy.filter_seconds = self.filter_seconds
        copy.overlay_seconds = self.overlay_seconds
        copy.write_seconds = self.write_seconds
        copy.confusion_seconds = self.confusion_seconds
        return copy

    def delta_since(self, before):
        """Return the counts and times added since ``before``."""
        delta = self.snapshot()
        for name in delta.class_names:
            delta.processed[name] -= before.processed.get(name, 0)
            delta.filtered[name] -= before.filtered.get(name, 0)
            delta.analysis_seconds[name] -= before.analysis_seconds.get(name, 0.0)
            delta.analysis_counts[name] -= before.analysis_counts.get(name, 0)
        delta.images_processed -= before.images_processed
        delta.windows_processed -= before.windows_processed
        delta.window_seconds -= before.window_seconds
        delta.inference_seconds -= before.inference_seconds
        delta.image_read_seconds -= before.image_read_seconds
        delta.scale_seconds -= before.scale_seconds
        delta.filter_seconds -= before.filter_seconds
        delta.overlay_seconds -= before.overlay_seconds
        delta.write_seconds -= before.write_seconds
        delta.confusion_seconds -= before.confusion_seconds
        return delta


def format_duration(seconds):
    return "%.4f s" % float(seconds)


def pixels_per_mm(um_per_px):
    """Convert micrometers per pixel into pixels per millimeter."""
    return 1000.0 / float(um_per_px)


def measurement_summary_lines(stats, total_seconds, um_per_px=None):
    """Return the count and timing lines printed at the end of a run."""
    filtered = sum(stats.filtered.values())
    combined = sum(stats.processed.values()) + filtered
    filter_bits = ", ".join("%s %d" % (name, stats.filtered.get(name, 0)) for name in stats.class_names)
    analysis_total = sum(stats.analysis_seconds.values())
    window_mean = stats.window_seconds / stats.windows_processed if stats.windows_processed else 0.0
    lines = ["Processing summary:"]
    if um_per_px is not None and um_per_px > 0:
        lines.append("Conversion factor: %.4f px/mm (%.6f um/px)" % (pixels_per_mm(um_per_px), um_per_px))
    lines.extend([
        "Cotton processed: %d" % stats.processed.get("cotton", 0),
        "Lumen processed: %d" % stats.processed.get("lumen", 0),
        "Filtered away: %d (%s)" % (filtered, filter_bits),
        "Combined total: %d" % combined,
        "Images processed: %d" % stats.images_processed,
        "Windows processed: %d" % stats.windows_processed,
        "Window processing time: %s" % format_duration(stats.window_seconds),
        "Window time per window: %s" % format_duration(window_mean),
        "Image read time: %s" % format_duration(stats.image_read_seconds),
        "Inference time: %s" % format_duration(stats.inference_seconds),
        "Contour filter time: %s" % format_duration(stats.filter_seconds),
        "Scale measurement time: %s" % format_duration(stats.scale_seconds),
        "Analysis processing time: %s" % format_duration(analysis_total),
    ])
    reported = ["cotton", "lumen"]
    reported.extend(
        name for name in stats.class_names if name not in reported and stats.analysis_counts.get(name, 0)
    )
    for class_name in reported:
        measured = stats.analysis_counts.get(class_name, 0)
        total = stats.analysis_seconds.get(class_name, 0.0)
        average = total / measured if measured else 0.0
        noun = "individual" if measured == 1 else "individuals"
        lines.append("Analysis time per %s: %s (%d %s)" % (class_name, format_duration(average), measured, noun))
    lines.append("Overlay time: %s" % format_duration(stats.overlay_seconds))
    lines.append("CSV write time: %s" % format_duration(stats.write_seconds))
    if stats.confusion_seconds > 0.0:
        lines.append("Confusion scoring time: %s" % format_duration(stats.confusion_seconds))
    lines.append("Total time: %s" % format_duration(total_seconds))
    return lines


def print_measurement_summary(stats, total_seconds, um_per_px=None, heading="Processing summary:"):
    print("")
    lines = measurement_summary_lines(stats, total_seconds, um_per_px=um_per_px)
    lines[0] = heading
    for line in lines:
        print(line)


def _require_image_dependencies():
    if cv2 is None or np is None:
        raise ModuleNotFoundError("fiber_analysis requires opencv-python and numpy to process images.")


def axis_length(axis):
    """Return Euclidean length for a two-point axis."""
    if axis is None:
        return 0.0
    return sqrt((float(axis[0][0]) - float(axis[1][0])) ** 2 + (float(axis[0][1]) - float(axis[1][1])) ** 2)


def contour_from_points(points):
    """Convert an Nx2 point sequence to an OpenCV contour (float32)."""
    _require_image_dependencies()
    return np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))


def as_draw_contour(contour):
    """Round a contour to int32 vertices for rasterization / drawing."""
    _require_image_dependencies()
    return np.round(np.asarray(contour, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))


def circularity_from_area_perimeter(area, perimeter):
    """Return 4*pi*area/perimeter^2, or None when undefined."""
    if area is None or perimeter is None or perimeter <= 0:
        return None
    return (4.0 * np.pi * float(area)) / (float(perimeter) ** 2)


def smoothed_contour(contour, epsilon_ratio=DEFAULT_SMOOTH_EPSILON_RATIO):
    """Return an approxPolyDP-smoothed closed contour."""
    _require_image_dependencies()
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    epsilon = max(float(epsilon_ratio) * float(perimeter), 1e-6)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx is None or len(approx) < 3:
        return np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    return approx


def contour_centroid(contour):
    """Return the spatial-moments centroid, falling back to bounding-box center."""
    _require_image_dependencies()
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-8:
        return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
    x, y, width, height = cv2.boundingRect(as_draw_contour(contour))
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def pixel_area_from_contour(image_shape, contour, conversion_factor=1.0):
    """Filled-mask pixel count scaled by conversion_factor^2."""
    mask = contour_mask(image_shape, contour)
    return float(cv2.countNonZero(mask)) * (float(conversion_factor) ** 2)


def find_longest_line(contour):
    """Find the longest point-to-point chord in a contour."""
    _require_image_dependencies()
    points = contour.reshape((-1, 2))
    best_length = 0.0
    best_axis = ((0.0, 0.0), (0.0, 0.0))

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


def measure_smart_grain(contour, epsilon_ratio=0.0005, perpendicular_tolerance_degrees=1.0):
    """Measure an approximate major axis and the longest near-perpendicular minor axis."""
    _require_image_dependencies()
    if contour is None or len(contour) < 2:
        zero_axis = ((0.0, 0.0), (0.0, 0.0))
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


def contour_mask(image_shape, contour):
    """Create a single-channel mask for a contour."""
    _require_image_dependencies()
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [as_draw_contour(contour)], 0, 255, -1)
    return mask


def is_inside(inner, outer, padding=1):
    """Return True when inner contour's bounding box sits inside outer contour's box."""
    _require_image_dependencies()
    ix, iy, iw, ih = cv2.boundingRect(as_draw_contour(inner))
    ox, oy, ow, oh = cv2.boundingRect(as_draw_contour(outer))
    return (
        ox <= ix + padding
        and oy <= iy + padding
        and ox + ow >= ix + iw - padding
        and oy + oh >= iy + ih - padding
    )


def find_cotton_lumen_pairs(cotton_contours, lumen_contours):
    """Pair each cotton contour with the first lumen contour contained by it."""
    lumen = list(lumen_contours)
    pairs = []

    for cotton in cotton_contours:
        lumen_match = next((candidate for candidate in lumen if is_inside(candidate, cotton)), None)
        pairs.append((cotton, lumen_match))

    return pairs


def conversion_factor_from_yellow_legend(
    image,
    legend_width_microns=40.0,
    yellow_lower_hsv=(25, 0, 0),
    yellow_upper_hsv=(36, 255, 255),
):
    """Estimate microns-per-pixel from the largest yellow legend.

    A missing or zero-width bar raises ValueError. Pass an explicit
    micrometers-per-pixel value instead of relying on the legend.
    """
    _require_image_dependencies()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array(yellow_lower_hsv), np.array(yellow_upper_hsv))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No yellow scale bar found. Pass micrometers per pixel instead of relying on the legend.")

    _, _, width, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if width <= 0:
        raise ValueError("Yellow scale bar has zero width. Pass micrometers per pixel instead of relying on the legend.")
    return float(legend_width_microns) / float(width)


def _format_number(value):
    if value is None:
        return "None"
    try:
        return format(float(value), ".6f")
    except (TypeError, ValueError):
        return str(value)


class FiberMeasurement(object):
    def __init__(self, filename, class_name, contour, conversion_factor=1.0, image_shape=None):
        self.filename = filename
        self.class_name = class_name
        self.contour = contour
        self.conversion_factor = conversion_factor
        self.image_shape = image_shape
        self.area = None
        self.pixel_area = None
        self.bounding_rect_area = None
        self.min_rect_area = None
        self.hull_area = None
        self.perimeter = None
        self.smoothed_perimeter = None
        self.is_convex = None
        self.circularity = None
        self.smoothed_circularity = None
        self.pixel_circularity = None
        self.min_axis = None
        self.max_axis = None
        self.min_axis_length = None
        self.max_axis_length = None
        self.aspect_ratio = None
        self.min_rect_orientation = None
        self.extent = None
        self.solidity = None
        self.equivalent_diameter = None
        self.ellipse_major_axis = None
        self.ellipse_minor_axis = None
        self.ellipse_orientation_angle = None
        self.eccentricity = None
        self.position = (-1.0, -1.0)
        self.centroid = (-1.0, -1.0)

    @classmethod
    def analyze(cls, filename, image, contour, class_name, conversion_factor=None, smooth_epsilon_ratio=DEFAULT_SMOOTH_EPSILON_RATIO, stats=None, full_metrics=False):
        _require_image_dependencies()
        factor = conversion_factor
        if factor is None:
            factor = conversion_factor_from_yellow_legend(image)

        measurement = cls(
            filename=filename,
            class_name=class_name,
            contour=contour,
            conversion_factor=factor,
            image_shape=None if image is None else image.shape,
        )
        started = time.perf_counter() if stats is not None else None
        measurement._analyze(smooth_epsilon_ratio=smooth_epsilon_ratio, full_metrics=full_metrics)
        if started is not None:
            stats.record_analysis(class_name, time.perf_counter() - started)
        return measurement

    def _analyze(self, smooth_epsilon_ratio=DEFAULT_SMOOTH_EPSILON_RATIO, full_metrics=False):
        _require_image_dependencies()
        contour = np.asarray(self.contour, dtype=np.float32).reshape((-1, 1, 2))
        self.contour = contour
        factor = self.conversion_factor

        self.area = cv2.contourArea(contour) * (factor**2)
        self.perimeter = cv2.arcLength(contour, True) * factor
        self.max_axis, self.min_axis = measure_smart_grain(contour)
        self.max_axis_length = axis_length(self.max_axis) * factor
        self.min_axis_length = axis_length(self.min_axis) * factor
        
        if not full_metrics:
            return
        draw_contour = as_draw_contour(contour)
        self.is_convex = bool(cv2.isContourConvex(draw_contour))

        x, y, width, height = cv2.boundingRect(draw_contour)
        self.bounding_rect_area = width * height * (factor**2)
        self.aspect_ratio = float(width) / float(height) if height else None

        min_rect = cv2.minAreaRect(contour)
        _, size, self.min_rect_orientation = min_rect
        self.min_rect_area = size[0] * size[1] * (factor**2)

        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        self.position = (float(cx), float(cy))
        self.centroid = contour_centroid(contour)

        if self.image_shape is not None:
            self.pixel_area = pixel_area_from_contour(self.image_shape, contour, conversion_factor=factor)
        else:
            self.pixel_area = None

        smooth = smoothed_contour(contour, epsilon_ratio=smooth_epsilon_ratio)
        self.smoothed_perimeter = cv2.arcLength(smooth, True) * factor

        hull = cv2.convexHull(contour)
        self.hull_area = cv2.contourArea(hull) * (factor**2)
        self.extent = self.area / self.bounding_rect_area if self.bounding_rect_area else None
        self.solidity = self.area / self.hull_area if self.hull_area else None
        self.equivalent_diameter = sqrt(4.0 * self.area / np.pi) if self.area else None
        self.circularity = circularity_from_area_perimeter(self.area, self.perimeter)
        self.smoothed_circularity = circularity_from_area_perimeter(self.area, self.smoothed_perimeter)
        self.pixel_circularity = circularity_from_area_perimeter(self.pixel_area, self.smoothed_perimeter)

        if len(hull) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(hull)
            self.ellipse_major_axis = max(axis_a, axis_b) * factor
            self.ellipse_minor_axis = min(axis_a, axis_b) * factor
            self.ellipse_orientation_angle = angle
            if self.ellipse_major_axis:
                axis_ratio = min(1.0, self.ellipse_minor_axis / self.ellipse_major_axis)
                self.eccentricity = sqrt(1.0 - axis_ratio * axis_ratio)
            else:
                self.eccentricity = None


    def to_csv_row(self, index):
        return [
            index,
            self.filename,
            self.class_name,
            _format_number(self.area),
            _format_number(self.pixel_area),
            _format_number(self.bounding_rect_area),
            _format_number(self.min_rect_area),
            _format_number(self.hull_area),
            _format_number(self.perimeter),
            _format_number(self.smoothed_perimeter),
            self.is_convex,
            _format_number(self.circularity),
            _format_number(self.smoothed_circularity),
            _format_number(self.pixel_circularity),
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
            _format_number(self.position[0]),
            _format_number(self.position[1]),
            _format_number(self.centroid[0]),
            _format_number(self.centroid[1]),
        ]


def write_measurements_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        for index, measurement in rows:
            writer.writerow(measurement.to_csv_row(index))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Measure contours in an OpenCV-readable mask image.")
    parser.add_argument("mask", type=Path, help="Binary or grayscale mask image.")
    parser.add_argument("--image", type=Path, help="Original image used for legend-based unit conversion.")
    parser.add_argument("--output-csv", type=Path, default=Path("fiber_measurements.csv"))
    parser.add_argument("--class-name", default="cotton", help="Class label to write to the CSV.")
    parser.add_argument("--min-area", type=float, default=0.0, help="Discard contours below this pixel area.")
    parser.add_argument("--full-metrics", action="store_true", help="Also compute morphology beyond area and perimeter. Those metrics are skipped unless this is set.")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    started = time.perf_counter()
    stats = MeasurementTiming((args.class_name,))
    _require_image_dependencies()
    read_started = time.perf_counter()
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Could not read mask: %s" % args.mask)

    image = cv2.imread(str(args.image)) if args.image else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if image is None:
        raise FileNotFoundError("Could not read image: %s" % args.image)
    stats.image_read_seconds += time.perf_counter() - read_started
    stats.images_processed = 1

    extract_started = time.perf_counter()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    stats.filter_seconds += time.perf_counter() - extract_started
    contour_areas = [(index, contour, cv2.contourArea(contour)) for index, contour in enumerate(contours)]
    kept = []
    for index, contour, area in contour_areas:
        if area < args.min_area:
            stats.record_filtered(args.class_name)
            continue
        kept.append((index, contour))
    conversion_factor = None
    if kept:
        scale_started = time.perf_counter()
        conversion_factor = conversion_factor_from_yellow_legend(image)
        stats.scale_seconds += time.perf_counter() - scale_started
    measurements = []
    for index, contour in kept:
        stats.record_processed(args.class_name)
        measurements.append(
            (
                index,
                FiberMeasurement.analyze(
                    str(args.mask),
                    image,
                    contour,
                    class_name=args.class_name,
                    conversion_factor=conversion_factor,
                    stats=stats,
                    full_metrics=args.full_metrics,
                ),
            )
        )

    write_started = time.perf_counter()
    write_measurements_csv(args.output_csv, measurements)
    stats.write_seconds += time.perf_counter() - write_started
    print("Wrote %s measurements to %s" % (len(measurements), args.output_csv))
    print_measurement_summary(
        stats,
        time.perf_counter() - started,
        um_per_px=conversion_factor,
        heading="Image summary: %s" % args.mask.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
