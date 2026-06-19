#!/usr/bin/env python
"""Run YOLO segmentation on fiber images and export cotton/lumen measurements."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
from pathlib import Path
import struct
from typing import Iterable, Optional, Sequence

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ModuleNotFoundError:
    cv2 = None
    np = None
    YOLO = None

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    Image = None
    ImageOps = None

from measure import (
    CSV_HEADER,
    FiberMeasurement,
    conversion_factor_from_yellow_legend,
    find_cotton_lumen_pairs,
)


DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_CLASS_NAMES = ("cotton", "lumen")
DEFAULT_COLORS = {
    "cotton": (0, 255, 0),
    "lumen": (0, 0, 255),
}
def _require_yolo_dependencies() -> None:
    if cv2 is None or np is None or YOLO is None:
        raise ModuleNotFoundError("yolo_fiber_analysis requires opencv-python, numpy, and ultralytics.")


def parse_extensions(values: Iterable[str]) -> tuple[str, ...]:
    extensions = []
    for value in values:
        suffix = value.lower()
        extensions.append(suffix if suffix.startswith(".") else f".{suffix}")
    return tuple(extensions)


def iter_images(image_dir: Path, extensions: tuple[str, ...], *, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in image_dir.glob(pattern) if path.is_file() and path.suffix.lower() in extensions)


def strip_png_text_chunks(image_path: Path) -> bytes:
    """Return PNG bytes without oversized text metadata chunks."""
    signature = b"\x89PNG\r\n\x1a\n"
    data = image_path.read_bytes()
    if not data.startswith(signature):
        return data

    cleaned = bytearray(signature)
    offset = len(signature)
    text_chunk_types = {b"tEXt", b"zTXt", b"iTXt"}

    while offset + 12 <= len(data):
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(data):
            raise ValueError(f"Invalid PNG chunk length in {image_path}")

        if chunk_type not in text_chunk_types:
            cleaned.extend(data[offset:chunk_end])

        offset = chunk_end
        if chunk_type == b"IEND":
            break

    return bytes(cleaned)


def read_image_with_pillow(image_path: Path) -> np.ndarray:
    if Image is None or ImageOps is None:
        raise ValueError(f"Could not read image with OpenCV, and Pillow is not installed: {image_path}")

    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            return np.asarray(pil_image)
    except (OSError, ValueError):
        if image_path.suffix.lower() != ".png":
            raise

    with Image.open(BytesIO(strip_png_text_chunks(image_path))) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        return np.asarray(pil_image)


def read_image_bgr(image_path: Path) -> np.ndarray:
    """Read an image as BGR, using Pillow when OpenCV rejects PNG metadata."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    rgb_image = read_image_with_pillow(image_path)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def class_name_for_index(model: YOLO, index: int, fallback_names: Sequence[str]) -> str:
    names = getattr(model, "names", None)
    if isinstance(names, dict) and index in names:
        return str(names[index]).strip().lower()
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index]).strip().lower()
    if index < len(fallback_names):
        return str(fallback_names[index]).strip().lower()
    return str(index)


def contour_from_mask_polygon(mask_polygon: np.ndarray) -> np.ndarray:
    return mask_polygon.astype(np.int32).reshape((-1, 1, 2))


def passes_area_filter(
    contour: np.ndarray,
    *,
    class_name: str,
    image_area: int,
    min_cotton_area: float,
    min_lumen_area: float,
    max_area_ratio: float,
) -> bool:
    area = cv2.contourArea(contour)
    if area > image_area * max_area_ratio:
        return False
    if class_name == "cotton":
        return area >= min_cotton_area
    if class_name == "lumen":
        return area >= min_lumen_area
    return False


def collect_contours(
    result,
    model: YOLO,
    *,
    class_names: Sequence[str],
    min_cotton_area: float,
    min_lumen_area: float,
    max_area_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    cotton: list[np.ndarray] = []
    lumen: list[np.ndarray] = []
    image = np.copy(result.orig_img)

    if result.masks is None or result.boxes is None:
        return cotton, lumen, image

    image_height, image_width = result.orig_shape[:2]
    image_area = image_height * image_width

    for index, mask_polygon in enumerate(result.masks.xy):
        class_index = int(result.boxes.cls[index].item())
        class_name = class_name_for_index(model, class_index, class_names)
        contour = contour_from_mask_polygon(mask_polygon)

        if not passes_area_filter(
            contour,
            class_name=class_name,
            image_area=image_area,
            min_cotton_area=min_cotton_area,
            min_lumen_area=min_lumen_area,
            max_area_ratio=max_area_ratio,
        ):
            continue

        if class_name == "cotton":
            cotton.append(contour)
        elif class_name == "lumen":
            lumen.append(contour)

    return cotton, lumen, image


def draw_overlay(
    image: np.ndarray,
    cotton_contours: Iterable[np.ndarray],
    lumen_contours: Iterable[np.ndarray],
    *,
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    for contour in cotton_contours:
        cv2.drawContours(overlay, [contour], -1, DEFAULT_COLORS["cotton"], thickness=cv2.FILLED)
    for contour in lumen_contours:
        cv2.drawContours(overlay, [contour], -1, DEFAULT_COLORS["lumen"], thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def translate_contour(contour: np.ndarray, dx: int, dy: int) -> np.ndarray:
    offset = np.array([[[dx, dy]]], dtype=contour.dtype)
    return contour + offset


def draw_measurement_visualization(
    image: np.ndarray,
    *,
    cotton_measurements: Sequence[tuple[int, FiberMeasurement]],
    lumen_measurements: Sequence[tuple[int, FiberMeasurement]],
    alpha: float,
) -> np.ndarray:
    cotton_contours = [measurement.contour for _, measurement in cotton_measurements]
    lumen_contours = [measurement.contour for _, measurement in lumen_measurements]
    visualization = draw_overlay(image, cotton_contours, lumen_contours, alpha=alpha)

    for object_index, measurement in cotton_measurements:
        cv2.drawContours(visualization, [measurement.contour], -1, DEFAULT_COLORS["cotton"], 2)
        if measurement.max_axis is not None:
            cv2.line(
                visualization,
                tuple(map(int, measurement.max_axis[0])),
                tuple(map(int, measurement.max_axis[1])),
                (0, 0, 255),
                1,
            )
        if measurement.min_axis is not None:
            cv2.line(
                visualization,
                tuple(map(int, measurement.min_axis[0])),
                tuple(map(int, measurement.min_axis[1])),
                (0, 255, 255),
                1,
            )
        x, y, _, _ = cv2.boundingRect(measurement.contour)
        cv2.putText(
            visualization,
            str(object_index),
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            DEFAULT_COLORS["cotton"],
            1,
            cv2.LINE_AA,
        )

    for object_index, measurement in lumen_measurements:
        cv2.drawContours(visualization, [measurement.contour], -1, DEFAULT_COLORS["lumen"], 2)
        x, y, _, _ = cv2.boundingRect(measurement.contour)
        cv2.putText(
            visualization,
            str(object_index),
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            DEFAULT_COLORS["lumen"],
            1,
            cv2.LINE_AA,
        )

    return visualization


def write_rows(path: Path, rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def predict_one(model: YOLO, image: np.ndarray, predict_options: dict[str, object]):
    results = model(image, **predict_options)
    if not results:
        return None
    return results[0]


def sliding_window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]

    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_sliding_windows(image: np.ndarray, window_size: int, stride: int):
    image_height, image_width = image.shape[:2]
    for y in sliding_window_starts(image_height, window_size, stride):
        for x in sliding_window_starts(image_width, window_size, stride):
            window = image[y : min(y + window_size, image_height), x : min(x + window_size, image_width)]
            valid_height, valid_width = window.shape[:2]
            if valid_height == window_size and valid_width == window_size:
                yield window, x, y, valid_width, valid_height
                continue

            padded = np.zeros((window_size, window_size, image.shape[2]), dtype=image.dtype)
            padded[:valid_height, :valid_width] = window
            yield padded, x, y, valid_width, valid_height


def contour_is_inside_valid_window(contour: np.ndarray, valid_width: int, valid_height: int) -> bool:
    x, y, width, height = cv2.boundingRect(contour)
    return x >= 0 and y >= 0 and x + width <= valid_width and y + height <= valid_height


def contour_touches_internal_window_edge(
    contour: np.ndarray,
    *,
    window_x: int,
    window_y: int,
    valid_width: int,
    valid_height: int,
    image_width: int,
    image_height: int,
    margin: int,
) -> bool:
    x, y, width, height = cv2.boundingRect(contour)
    touches_left = window_x > 0 and x <= margin
    touches_top = window_y > 0 and y <= margin
    touches_right = window_x + valid_width < image_width and x + width >= valid_width - margin
    touches_bottom = window_y + valid_height < image_height and y + height >= valid_height - margin
    return touches_left or touches_top or touches_right or touches_bottom


def bounding_rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = cv2.boundingRect(a)
    bx, by, bw, bh = cv2.boundingRect(b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def dedupe_contours_by_iou(contours: Sequence[np.ndarray], *, iou_threshold: float) -> tuple[list[np.ndarray], int]:
    if iou_threshold <= 0:
        return list(contours), 0

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    kept: list[np.ndarray] = []
    removed = 0

    for contour in sorted_contours:
        if any(bounding_rect_iou(contour, kept_contour) >= iou_threshold for kept_contour in kept):
            removed += 1
            continue
        kept.append(contour)

    return kept, removed


def filter_large_contour_outliers(
    contours: Sequence[np.ndarray],
    *,
    stddevs: float,
    min_samples: int,
) -> tuple[list[np.ndarray], int, Optional[float]]:
    if stddevs <= 0 or len(contours) < min_samples:
        return list(contours), 0, None

    areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float64)
    area_std = float(np.std(areas))
    if area_std <= 1e-8:
        return list(contours), 0, None

    threshold = float(np.mean(areas) + stddevs * area_std)
    filtered = [contour for contour, area in zip(contours, areas) if area <= threshold]
    return filtered, len(contours) - len(filtered), threshold


def collect_sliding_window_contours(
    model: YOLO,
    *,
    image: np.ndarray,
    class_names: Sequence[str],
    predict_options: dict[str, object],
    window_size: int,
    window_stride: int,
    window_edge_margin: int,
    min_cotton_area: float,
    min_lumen_area: float,
    max_area_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    cotton_contours: list[np.ndarray] = []
    lumen_contours: list[np.ndarray] = []
    ignored_edge_detections = 0
    image_height, image_width = image.shape[:2]

    for window, window_x, window_y, valid_width, valid_height in iter_sliding_windows(
        image,
        window_size,
        window_stride,
    ):
        result = predict_one(model, window, predict_options)
        if result is None:
            continue

        window_cotton, window_lumen, _ = collect_contours(
            result,
            model,
            class_names=class_names,
            min_cotton_area=min_cotton_area,
            min_lumen_area=min_lumen_area,
            max_area_ratio=max_area_ratio,
        )

        for contour in window_cotton:
            if not contour_is_inside_valid_window(contour, valid_width, valid_height):
                continue
            if contour_touches_internal_window_edge(
                contour,
                window_x=window_x,
                window_y=window_y,
                valid_width=valid_width,
                valid_height=valid_height,
                image_width=image_width,
                image_height=image_height,
                margin=window_edge_margin,
            ):
                ignored_edge_detections += 1
                continue
            cotton_contours.append(translate_contour(contour, window_x, window_y))

        for contour in window_lumen:
            if not contour_is_inside_valid_window(contour, valid_width, valid_height):
                continue
            if contour_touches_internal_window_edge(
                contour,
                window_x=window_x,
                window_y=window_y,
                valid_width=valid_width,
                valid_height=valid_height,
                image_width=image_width,
                image_height=image_height,
                margin=window_edge_margin,
            ):
                ignored_edge_detections += 1
                continue
            lumen_contours.append(translate_contour(contour, window_x, window_y))

    return cotton_contours, lumen_contours, ignored_edge_detections


def analyze_image_sliding_window(
    model: YOLO,
    *,
    image: np.ndarray,
    image_path: Path,
    output_dir: Path,
    class_names: Sequence[str],
    predict_options: dict[str, object],
    window_size: int,
    window_stride: int,
    window_edge_margin: int,
    dedupe_iou: float,
    large_contour_stddevs: float,
    large_contour_min_samples: int,
    min_cotton_area: float,
    min_lumen_area: float,
    max_area_ratio: float,
    overlay_alpha: float,
    save_overlays: bool,
    verbose: bool,
    start_index: int,
) -> tuple[list[list[object]], int]:
    conversion_factor = conversion_factor_from_yellow_legend(image)
    if verbose:
        print(f"  legend conversion factor: {conversion_factor:.6f}")
    cotton_contours, lumen_contours, ignored_edge_detections = collect_sliding_window_contours(
        model,
        image=image,
        class_names=class_names,
        predict_options=predict_options,
        window_size=window_size,
        window_stride=window_stride,
        window_edge_margin=window_edge_margin,
        min_cotton_area=min_cotton_area,
        min_lumen_area=min_lumen_area,
        max_area_ratio=max_area_ratio,
    )

    raw_cotton_count = len(cotton_contours)
    raw_lumen_count = len(lumen_contours)
    cotton_contours, cotton_dedupe_removed = dedupe_contours_by_iou(cotton_contours, iou_threshold=dedupe_iou)
    lumen_contours, lumen_dedupe_removed = dedupe_contours_by_iou(lumen_contours, iou_threshold=dedupe_iou)
    cotton_contours, cotton_removed, cotton_threshold = filter_large_contour_outliers(
        cotton_contours,
        stddevs=large_contour_stddevs,
        min_samples=large_contour_min_samples,
    )
    lumen_contours, lumen_removed, lumen_threshold = filter_large_contour_outliers(
        lumen_contours,
        stddevs=large_contour_stddevs,
        min_samples=large_contour_min_samples,
    )
    if verbose:
        print(
            f"  sliding windows: cotton {raw_cotton_count} raw / {cotton_dedupe_removed} duplicate / "
            f"{len(cotton_contours)} kept; lumen {raw_lumen_count} raw / "
            f"{lumen_dedupe_removed} duplicate / {len(lumen_contours)} kept; "
            f"{ignored_edge_detections} edge detections ignored"
        )
        print(
            f"  large-contour filter: cotton removed {cotton_removed}"
            f"{f' above {cotton_threshold:.1f}px' if cotton_threshold else ''}; "
            f"lumen removed {lumen_removed}"
            f"{f' above {lumen_threshold:.1f}px' if lumen_threshold else ''}"
        )

    rows: list[list[object]] = []
    cotton_measurements: list[tuple[int, FiberMeasurement]] = []
    lumen_measurements: list[tuple[int, FiberMeasurement]] = []
    object_index = start_index
    pairs = find_cotton_lumen_pairs(cotton_contours, lumen_contours)

    for cotton_contour, lumen_contour in pairs:
        cotton = FiberMeasurement.analyze(
            str(image_path),
            image=image,
            contour=cotton_contour,
            class_name="cotton",
            conversion_factor=conversion_factor,
        )
        rows.append(cotton.to_csv_row(object_index))
        cotton_measurements.append((object_index, cotton))

        if lumen_contour is not None:
            lumen = FiberMeasurement.analyze(
                str(image_path),
                image=image,
                contour=lumen_contour,
                class_name="lumen",
                conversion_factor=conversion_factor,
            )
            rows.append(lumen.to_csv_row(object_index))
            lumen_measurements.append((object_index, lumen))

        object_index += 1

    if save_overlays:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        visualization = draw_measurement_visualization(
            image,
            cotton_measurements=cotton_measurements,
            lumen_measurements=lumen_measurements,
            alpha=overlay_alpha,
        )
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}_overlay.png"), visualization)

    return rows, object_index


def run_analysis(args: argparse.Namespace) -> int:
    _require_yolo_dependencies()
    if args.window_size <= 0:
        raise ValueError("--window-size must be greater than 0")
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be greater than 0")
    if args.window_edge_margin < 0:
        raise ValueError("--window-edge-margin must be greater than or equal to 0")
    if args.dedupe_iou < 0 or args.dedupe_iou > 1:
        raise ValueError("--dedupe-iou must be between 0 and 1")
    if args.large_contour_stddevs < 0:
        raise ValueError("--large-contour-stddevs must be greater than or equal to 0")
    if args.large_contour_min_samples < 1:
        raise ValueError("--large-contour-min-samples must be greater than or equal to 1")

    model = YOLO(str(args.model))
    extensions = parse_extensions(args.extensions)
    images = iter_images(args.image_dir, extensions, recursive=args.recursive)

    if not images:
        raise FileNotFoundError(f"No images found in {args.image_dir} with extensions {', '.join(extensions)}")

    output_csv = args.output_csv or args.output_dir / "fiber_measurements.csv"
    all_rows: list[list[object]] = []
    object_index = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_number, image_path in enumerate(images, start=1):
        print(f"[{image_number}/{len(images)}] {image_path}")
        image = read_image_bgr(image_path)
        predict_options = {
            "show_labels": False,
            "line_width": args.line_width,
            "conf": args.confidence,
            "iou": args.iou,
            "verbose": args.verbose,
            "imgsz": args.window_size,
        }
        if args.device is not None:
            predict_options["device"] = args.device

        rows, object_index = analyze_image_sliding_window(
            model,
            image=image,
            image_path=image_path,
            output_dir=args.output_dir,
            class_names=args.class_names,
            predict_options=predict_options,
            window_size=args.window_size,
            window_stride=args.window_stride,
            window_edge_margin=args.window_edge_margin,
            dedupe_iou=args.dedupe_iou,
            large_contour_stddevs=args.large_contour_stddevs,
            large_contour_min_samples=args.large_contour_min_samples,
            min_cotton_area=args.min_cotton_area,
            min_lumen_area=args.min_lumen_area,
            max_area_ratio=args.max_area_ratio,
            overlay_alpha=args.overlay_alpha,
            save_overlays=args.save_overlays,
            verbose=args.verbose,
            start_index=object_index,
        )
        all_rows.extend(rows)

    write_rows(output_csv, all_rows)
    print(f"Wrote {len(all_rows)} rows for {object_index} cotton objects to {output_csv}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to YOLO segmentation weights.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing images to analyze.")
    parser.add_argument("--output-dir", type=Path, default=Path("fiber_output"), help="Directory for analysis outputs.")
    parser.add_argument("--output-csv", type=Path, help="Aggregate CSV path. Defaults to output-dir/fiber_measurements.csv.")
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS, help="Image file extensions to include.")
    parser.add_argument("--recursive", action="store_true", help="Search image-dir recursively.")
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES, help="Fallback class names by YOLO index.")
    parser.add_argument("--min-cotton-area", type=float, default=100.0, help="Minimum cotton contour area in pixels.")
    parser.add_argument("--min-lumen-area", type=float, default=50.0, help="Minimum lumen contour area in pixels.")
    parser.add_argument("--max-area-ratio", type=float, default=0.95, help="Discard masks larger than this fraction of window area.")
    parser.add_argument("--window-size", type=int, default=512, help="Sliding-window square size for inference.")
    parser.add_argument("--window-stride", type=int, default=256, help="Sliding-window stride in pixels.")
    parser.add_argument("--window-edge-margin", type=int, default=2, help="Ignore detections this close to an internal window edge.")
    parser.add_argument("--dedupe-iou", type=float, default=0.5, help="Remove duplicate same-class detections with bounding-box IoU at or above this value.")
    parser.add_argument("--large-contour-stddevs", type=float, default=3.0, help="Remove contours larger than mean area plus this many standard deviations. Use 0 to disable.")
    parser.add_argument("--large-contour-min-samples", type=int, default=6, help="Minimum contour count before applying the large-contour outlier filter.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    parser.add_argument("--device", help="YOLO device, e.g. cpu, 0, or cuda:0.")
    parser.add_argument("--line-width", type=int, default=1, help="Line width for YOLO-rendered results.")
    parser.add_argument("--save-overlays", action="store_true", default=True, help="Save one merged cotton/lumen overlay for each source image.")
    parser.add_argument("--no-save-overlays", action="store_false", dest="save_overlays", help="Disable merged overlay image output.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Overlay opacity.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose YOLO logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
