"""Run YOLO segmentation on fiber images and export cotton/lumen measurements."""
import argparse
import csv
import math
import time
from io import BytesIO
from pathlib import Path
import struct
try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None
try:
    from ultralytics import YOLO
except (ImportError, ModuleNotFoundError):
    YOLO = None
try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    Image = None
    ImageOps = None
from measure import (
    CSV_HEADER,
    FiberMeasurement,
    MeasurementTiming,
    as_draw_contour,
    conversion_factor_from_yellow_legend,
    find_cotton_lumen_pairs,
    print_measurement_summary,
)
DEFAULT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
DEFAULT_CLASS_NAMES = ('cotton', 'lumen')
DEFAULT_COLORS = {'cotton': (0, 255, 0), 'lumen': (0, 0, 255)}
# Older val_confusion imports used these names before timing moved into measure.py.
ProcessingStats = MeasurementTiming
print_processing_summary = print_measurement_summary

def _require_yolo_dependencies():
    if cv2 is None or np is None or YOLO is None:
        raise ModuleNotFoundError('yolo_fiber_analysis requires opencv-python, numpy, and ultralytics.')

def parse_extensions(values):
    extensions = []
    for value in values:
        suffix = value.lower()
        extensions.append(suffix if suffix.startswith('.') else f'.{suffix}')
    return tuple(extensions)

def iter_images(image_dir, extensions, *, recursive):
    pattern = '**/*' if recursive else '*'
    return sorted((path for path in image_dir.glob(pattern) if path.is_file() and path.suffix.lower() in extensions))

def strip_png_text_chunks(image_path):
    """Return PNG bytes without oversized text metadata chunks."""
    signature = b'\x89PNG\r\n\x1a\n'
    data = image_path.read_bytes()
    if not data.startswith(signature):
        return data
    cleaned = bytearray(signature)
    offset = len(signature)
    text_chunk_types = {b'tEXt', b'zTXt', b'iTXt'}
    while offset + 12 <= len(data):
        chunk_length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(data):
            raise ValueError(f'Invalid PNG chunk length in {image_path}')
        if chunk_type not in text_chunk_types:
            cleaned.extend(data[offset:chunk_end])
        offset = chunk_end
        if chunk_type == b'IEND':
            break
    return bytes(cleaned)

def read_image_with_pillow(image_path):
    if Image is None or ImageOps is None:
        raise ValueError(f'Could not read image with OpenCV, and Pillow is not installed: {image_path}')
    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
            return np.asarray(pil_image)
    except (OSError, ValueError):
        if image_path.suffix.lower() != '.png':
            raise
    with Image.open(BytesIO(strip_png_text_chunks(image_path))) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        return np.asarray(pil_image)

def read_image_bgr(image_path):
    """Read an image as BGR, using Pillow when OpenCV rejects PNG metadata."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    rgb_image = read_image_with_pillow(image_path)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

def class_name_for_index(model, index, fallback_names):
    names = getattr(model, 'names', None)
    if isinstance(names, dict) and index in names:
        return str(names[index]).strip().lower()
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index]).strip().lower()
    if index < len(fallback_names):
        return str(fallback_names[index]).strip().lower()
    return str(index)

def contour_from_mask_polygon(mask_polygon):
    """Keep subpixel YOLO polygon vertices for measurement."""
    return np.asarray(mask_polygon, dtype=np.float32).reshape((-1, 1, 2))

def passes_area_filter(contour, *, class_name, image_area, min_cotton_area, min_lumen_area, max_area_ratio):
    area = cv2.contourArea(contour)
    if area > image_area * max_area_ratio:
        return False
    if class_name == 'cotton':
        return area >= min_cotton_area
    if class_name == 'lumen':
        return area >= min_lumen_area
    return False

def collect_contours(result, model, *, class_names, min_cotton_area, min_lumen_area, max_area_ratio, stats=None):
    cotton = []
    lumen = []
    image = np.copy(result.orig_img)
    if result.boxes is None or result.masks is None:
        return (cotton, lumen, image)
    image_height, image_width = result.orig_shape[:2]
    image_area = image_height * image_width
    for index, mask_polygon in enumerate(result.masks.xy):
        class_index = int(result.boxes.cls[index].item())
        class_name = class_name_for_index(model, class_index, class_names)
        contour = contour_from_mask_polygon(mask_polygon)
        if not passes_area_filter(contour, class_name=class_name, image_area=image_area, min_cotton_area=min_cotton_area, min_lumen_area=min_lumen_area, max_area_ratio=max_area_ratio):
            if stats is not None:
                stats.record_filtered(class_name)
            continue
        if class_name == 'cotton':
            cotton.append(contour)
        elif class_name == 'lumen':
            lumen.append(contour)
    return (cotton, lumen, image)

def draw_overlay(image, cotton_contours, lumen_contours, *, alpha):
    overlay = image.copy()
    for contour in cotton_contours:
        cv2.drawContours(overlay, [as_draw_contour(contour)], -1, DEFAULT_COLORS['cotton'], thickness=cv2.FILLED)
    for contour in lumen_contours:
        cv2.drawContours(overlay, [as_draw_contour(contour)], -1, DEFAULT_COLORS['lumen'], thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

def translate_contour(contour, dx, dy):
    offset = np.array([[[dx, dy]]], dtype=np.float32)
    return np.asarray(contour, dtype=np.float32) + offset

def draw_measurement_visualization(image, *, cotton_measurements, lumen_measurements, alpha):
    cotton_contours = [measurement.contour for _, measurement in cotton_measurements]
    lumen_contours = [measurement.contour for _, measurement in lumen_measurements]
    visualization = draw_overlay(image, cotton_contours, lumen_contours, alpha=alpha)
    for object_index, measurement in cotton_measurements:
        draw_contour = as_draw_contour(measurement.contour)
        cv2.drawContours(visualization, [draw_contour], -1, DEFAULT_COLORS['cotton'], 2)
        if measurement.max_axis is not None:
            cv2.line(visualization, tuple(map(int, measurement.max_axis[0])), tuple(map(int, measurement.max_axis[1])), (0, 0, 255), 1)
        if measurement.min_axis is not None:
            cv2.line(visualization, tuple(map(int, measurement.min_axis[0])), tuple(map(int, measurement.min_axis[1])), (0, 255, 255), 1)
        x, y, _, _ = cv2.boundingRect(draw_contour)
        cv2.putText(visualization, str(object_index), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEFAULT_COLORS['cotton'], 1, cv2.LINE_AA)
    for object_index, measurement in lumen_measurements:
        draw_contour = as_draw_contour(measurement.contour)
        cv2.drawContours(visualization, [draw_contour], -1, DEFAULT_COLORS['lumen'], 2)
        x, y, _, _ = cv2.boundingRect(draw_contour)
        cv2.putText(visualization, str(object_index), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEFAULT_COLORS['lumen'], 1, cv2.LINE_AA)
    return visualization

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

def predict_one(model, image, predict_options):
    results = model(image, **predict_options)
    if not results:
        return None
    return results[0]

def sync_inference_device(device):
    """Wait for queued GPU work so inference timing includes the forward pass."""
    if device is not None and str(device).strip().lower() == 'cpu':
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def sliding_window_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def iter_sliding_windows(image, window_size, stride):
    image_height, image_width = image.shape[:2]
    for y in sliding_window_starts(image_height, window_size, stride):
        for x in sliding_window_starts(image_width, window_size, stride):
            window = image[y:min(y + window_size, image_height), x:min(x + window_size, image_width)]
            valid_height, valid_width = window.shape[:2]
            if valid_height == window_size and valid_width == window_size:
                yield (window, x, y, valid_width, valid_height)
                continue
            padded = np.zeros((window_size, window_size, image.shape[2]), dtype=image.dtype)
            padded[:valid_height, :valid_width] = window
            yield (padded, x, y, valid_width, valid_height)

def contour_is_inside_valid_window(contour, valid_width, valid_height):
    x, y, width, height = cv2.boundingRect(contour)
    return x >= 0 and y >= 0 and (x + width <= valid_width) and (y + height <= valid_height)

def contour_touches_window_or_image_edge(contour, *, valid_width, valid_height, margin):
    """Return True when a detection is clipped by a window seam or the composite border.

    The valid rectangle is the real image area inside this crop. On an interior
    window those edges are seams between crops. On a window that sits against
    the image, the same edges are the composite border. Both cuts leave a
    partial fiber, so neither is measured.
    """
    x, y, width, height = cv2.boundingRect(contour)
    touches_left = x <= margin
    touches_top = y <= margin
    touches_right = x + width >= valid_width - margin
    touches_bottom = y + height >= valid_height - margin
    return touches_left or touches_top or touches_right or touches_bottom

def bounding_rect_iou(a, b):
    ax, ay, aw, ah = cv2.boundingRect(a)
    bx, by, bw, bh = cv2.boundingRect(b)
    ax2, ay2 = (ax + aw, ay + ah)
    bx2, by2 = (bx + bw, by + bh)
    ix1, iy1 = (max(ax, bx), max(ay, by))
    ix2, iy2 = (min(ax2, bx2), min(ay2, by2))
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0

def dedupe_contours_by_iou(contours, *, iou_threshold):
    if iou_threshold <= 0:
        return (list(contours), 0)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    kept = []
    removed = 0
    for contour in sorted_contours:
        if any((bounding_rect_iou(contour, kept_contour) >= iou_threshold for kept_contour in kept)):
            removed += 1
            continue
        kept.append(contour)
    return (kept, removed)

def filter_large_contour_outliers(contours, *, stddevs, min_samples):
    if stddevs <= 0 or len(contours) < min_samples:
        return (list(contours), 0, None)
    areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float64)
    area_std = float(np.std(areas))
    if area_std <= 1e-08:
        return (list(contours), 0, None)
    threshold = float(np.mean(areas) + stddevs * area_std)
    filtered = [contour for contour, area in zip(contours, areas) if area <= threshold]
    return (filtered, len(contours) - len(filtered), threshold)

def postprocess_class_contours(contours, *, dedupe_iou, large_contour_stddevs, large_contour_min_samples, class_name, stats=None):
    """Drop duplicate and oversized contours. Record kept and removed counts."""
    started = time.perf_counter() if stats is not None else None
    contours, dedupe_removed = dedupe_contours_by_iou(contours, iou_threshold=dedupe_iou)
    contours, large_removed, threshold = filter_large_contour_outliers(contours, stddevs=large_contour_stddevs, min_samples=large_contour_min_samples)
    if stats is not None:
        stats.filter_seconds += time.perf_counter() - started
        stats.record_filtered(class_name, dedupe_removed + large_removed)
        stats.record_processed(class_name, len(contours))
    return (contours, dedupe_removed, large_removed, threshold)

def collect_sliding_window_contours(model, *, image, class_names, predict_options, window_size, window_stride, window_edge_margin, min_cotton_area, min_lumen_area, max_area_ratio, stats=None):
    cotton_contours = []
    lumen_contours = []
    ignored_edge_detections = 0
    kept = {'cotton': cotton_contours, 'lumen': lumen_contours}
    for window, window_x, window_y, valid_width, valid_height in iter_sliding_windows(image, window_size, window_stride):
        window_started = time.perf_counter() if stats is not None else None
        infer_started = time.perf_counter() if stats is not None else None
        result = predict_one(model, window, predict_options)
        if infer_started is not None:
            sync_inference_device(predict_options.get('device'))
            stats.inference_seconds += time.perf_counter() - infer_started
        if result is None:
            if window_started is not None:
                stats.record_window(time.perf_counter() - window_started)
            continue
        window_cotton, window_lumen, _ = collect_contours(result, model, class_names=class_names, min_cotton_area=min_cotton_area, min_lumen_area=min_lumen_area, max_area_ratio=max_area_ratio, stats=stats)
        window_contours = {'cotton': window_cotton, 'lumen': window_lumen}
        for class_name in ('cotton', 'lumen'):
            for contour in window_contours[class_name]:
                if not contour_is_inside_valid_window(contour, valid_width, valid_height):
                    if stats is not None:
                        stats.record_filtered(class_name)
                    continue
                if contour_touches_window_or_image_edge(contour, valid_width=valid_width, valid_height=valid_height, margin=window_edge_margin):
                    ignored_edge_detections += 1
                    if stats is not None:
                        stats.record_filtered(class_name)
                    continue
                kept[class_name].append(translate_contour(contour, window_x, window_y))
        if window_started is not None:
            stats.record_window(time.perf_counter() - window_started)
    return (cotton_contours, lumen_contours, ignored_edge_detections)

def analyze_image_sliding_window(model, *, image, image_path, output_dir, class_names, predict_options, window_size, window_stride, window_edge_margin, dedupe_iou, large_contour_stddevs, large_contour_min_samples, min_cotton_area, min_lumen_area, max_area_ratio, overlay_alpha, save_overlays, verbose, start_index, um_per_px, stats=None, full_metrics=False):
    scale_started = time.perf_counter() if stats is not None else None
    if um_per_px is not None:
        conversion_factor = um_per_px
        scale_source = 'parameter'
    else:
        try:
            conversion_factor = conversion_factor_from_yellow_legend(image)
        except ValueError as exc:
            raise ValueError(f'{image_path}: {exc}') from exc
        scale_source = 'yellow legend'
    if scale_started is not None:
        stats.scale_seconds += time.perf_counter() - scale_started
    if verbose:
        print(f'  conversion factor: {conversion_factor:.6f} um/px from {scale_source}')
    cotton_contours, lumen_contours, ignored_edge_detections = collect_sliding_window_contours(model, image=image, class_names=class_names, predict_options=predict_options, window_size=window_size, window_stride=window_stride, window_edge_margin=window_edge_margin, min_cotton_area=min_cotton_area, min_lumen_area=min_lumen_area, max_area_ratio=max_area_ratio, stats=stats)
    raw_cotton_count = len(cotton_contours)
    raw_lumen_count = len(lumen_contours)
    cotton_contours, cotton_dedupe_removed, cotton_removed, cotton_threshold = postprocess_class_contours(cotton_contours, dedupe_iou=dedupe_iou, large_contour_stddevs=large_contour_stddevs, large_contour_min_samples=large_contour_min_samples, class_name='cotton', stats=stats)
    lumen_contours, lumen_dedupe_removed, lumen_removed, lumen_threshold = postprocess_class_contours(lumen_contours, dedupe_iou=dedupe_iou, large_contour_stddevs=large_contour_stddevs, large_contour_min_samples=large_contour_min_samples, class_name='lumen', stats=stats)
    if verbose:
        print(f'  sliding windows: cotton {raw_cotton_count} raw / {cotton_dedupe_removed} duplicate / {len(cotton_contours)} kept; lumen {raw_lumen_count} raw / {lumen_dedupe_removed} duplicate / {len(lumen_contours)} kept; {ignored_edge_detections} edge detections ignored')
        print(f"  large-contour filter: cotton removed {cotton_removed}{(f' above {cotton_threshold:.1f}px' if cotton_threshold else '')}; lumen removed {lumen_removed}{(f' above {lumen_threshold:.1f}px' if lumen_threshold else '')}")
    rows = []
    cotton_measurements = []
    lumen_measurements = []
    object_index = start_index
    pairs = find_cotton_lumen_pairs(cotton_contours, lumen_contours)
    for cotton_contour, lumen_contour in pairs:
        cotton = FiberMeasurement.analyze(str(image_path), image=image, contour=cotton_contour, class_name='cotton', conversion_factor=conversion_factor, stats=stats, full_metrics=full_metrics)
        rows.append(cotton.to_csv_row(object_index))
        cotton_measurements.append((object_index, cotton))
        if lumen_contour is not None:
            lumen = FiberMeasurement.analyze(str(image_path), image=image, contour=lumen_contour, class_name='lumen', conversion_factor=conversion_factor, stats=stats, full_metrics=full_metrics)
            rows.append(lumen.to_csv_row(object_index))
            lumen_measurements.append((object_index, lumen))
        object_index += 1
    if save_overlays:
        overlay_started = time.perf_counter() if stats is not None else None
        overlay_dir = output_dir / 'overlays'
        overlay_dir.mkdir(parents=True, exist_ok=True)
        visualization = draw_measurement_visualization(image, cotton_measurements=cotton_measurements, lumen_measurements=lumen_measurements, alpha=overlay_alpha)
        cv2.imwrite(str(overlay_dir / f'{image_path.stem}_overlay.png'), visualization)
        if overlay_started is not None:
            stats.overlay_seconds += time.perf_counter() - overlay_started
    return (rows, object_index, cotton_contours, lumen_contours, conversion_factor)

def label_path_for_image(image_path, image_dir, labels_dir):
    """Return the YOLO label file for an image, or None when no label root exists."""
    if labels_dir is not None:
        relative = image_path.resolve().relative_to(image_dir.resolve())
        return (labels_dir / relative).with_suffix('.txt')
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == 'images':
            parts[index] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return None

def read_yolo_contours(label_path, image_width, image_height, class_names):
    """Load YOLO polygon labels as contours in pixel coordinates, binned by class."""
    grouped = {name: [] for name in class_names}
    text = label_path.read_text(encoding='utf-8')
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        class_index = int(float(parts[0]))
        if class_index < 0 or class_index >= len(class_names):
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size < 6 or coords.size % 2 != 0:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= image_width
        points[:, 1] *= image_height
        contour = points.reshape(-1, 1, 2)
        if cv2.contourArea(contour) <= 0:
            continue
        grouped[class_names[class_index]].append(contour)
    return grouped

def filter_boundary_contours(contours, *, class_name, image_shape, edge_margin, min_cotton_area, min_lumen_area, max_area_ratio):
    """Drop contours on the image border and contours outside the area limits.

    Predictions already skip a window seam or the composite border. Ground truth
    on that same border is dropped here so a clipped fiber is not a false negative.
    """
    image_height, image_width = image_shape[:2]
    image_area = float(image_height * image_width)
    kept = []
    skipped_edge = 0
    for contour in contours:
        if contour_touches_window_or_image_edge(contour, valid_width=image_width, valid_height=image_height, margin=edge_margin):
            skipped_edge += 1
            continue
        if not passes_area_filter(contour, class_name=class_name, image_area=image_area, min_cotton_area=min_cotton_area, min_lumen_area=min_lumen_area, max_area_ratio=max_area_ratio):
            continue
        kept.append(contour)
    return (kept, skipped_edge)

def contour_centroid(contour):
    moments = cv2.moments(contour)
    if moments['m00'] <= 1e-06:
        x, y, width, height = cv2.boundingRect(contour)
        return (float(x) + width / 2.0, float(y) + height / 2.0)
    return (float(moments['m10'] / moments['m00']), float(moments['m01'] / moments['m00']))

def association_reach(contour, tolerance):
    """How far a shifted fiber may sit and still be the same object.

    Half an equivalent radius covers a lumen that moved enough to wreck mask
    IoU while keeping the same area. The boundary tolerance is the floor.
    """
    area = float(cv2.contourArea(contour))
    radius = math.sqrt(area / math.pi) if area > 0 else 0.0
    return max(float(tolerance), 0.5 * radius)

def contours_correspond(prediction, ground_truth, tolerance):
    """True when the contours describe the same fiber location."""
    reach = max(association_reach(prediction, tolerance), association_reach(ground_truth, tolerance))
    pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(prediction)
    gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(ground_truth)
    separated_x = pred_x - reach > gt_x + gt_w or gt_x - reach > pred_x + pred_w
    separated_y = pred_y - reach > gt_y + gt_h or gt_y - reach > pred_y + pred_h
    if separated_x or separated_y:
        return False
    pred_draw = as_draw_contour(prediction)
    gt_draw = as_draw_contour(ground_truth)
    if len(pred_draw) < 3 or len(gt_draw) < 3:
        return False
    pred_centroid = contour_centroid(prediction)
    gt_centroid = contour_centroid(ground_truth)
    pred_inside = cv2.pointPolygonTest(gt_draw, pred_centroid, True) >= -association_reach(ground_truth, tolerance)
    gt_inside = cv2.pointPolygonTest(pred_draw, gt_centroid, True) >= -association_reach(prediction, tolerance)
    return pred_inside or gt_inside

def centroid_distance(prediction, ground_truth):
    pred_x, pred_y = contour_centroid(prediction)
    gt_x, gt_y = contour_centroid(ground_truth)
    return math.hypot(pred_x - gt_x, pred_y - gt_y)

def boundary_f1(prediction, ground_truth, tolerance):
    """F1 of the two contours' boundaries within ``tolerance`` pixels."""
    pred_draw = as_draw_contour(prediction)
    gt_draw = as_draw_contour(ground_truth)
    if len(pred_draw) < 2 or len(gt_draw) < 2:
        return 0.0
    pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(pred_draw)
    gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(gt_draw)
    pad = int(math.ceil(tolerance)) + 1
    origin_x = min(pred_x, gt_x) - pad
    origin_y = min(pred_y, gt_y) - pad
    width = max(pred_x + pred_w, gt_x + gt_w) + pad - origin_x
    height = max(pred_y + pred_h, gt_y + gt_h) + pad - origin_y
    if width <= 1 or height <= 1:
        return 0.0
    pred_boundary = np.zeros((height, width), dtype=np.uint8)
    gt_boundary = np.zeros((height, width), dtype=np.uint8)
    offset = np.array([[[origin_x, origin_y]]], dtype=np.int32)
    cv2.drawContours(pred_boundary, [pred_draw - offset], -1, 1, thickness=1)
    cv2.drawContours(gt_boundary, [gt_draw - offset], -1, 1, thickness=1)
    pred_count = int(np.count_nonzero(pred_boundary))
    gt_count = int(np.count_nonzero(gt_boundary))
    if pred_count == 0 or gt_count == 0:
        return 0.0
    distance_to_pred = cv2.distanceTransform(np.where(pred_boundary > 0, np.uint8(0), np.uint8(255)), cv2.DIST_L2, 3)
    distance_to_gt = cv2.distanceTransform(np.where(gt_boundary > 0, np.uint8(0), np.uint8(255)), cv2.DIST_L2, 3)
    precision = float(np.count_nonzero(distance_to_gt[pred_boundary > 0] <= tolerance)) / pred_count
    recall = float(np.count_nonzero(distance_to_pred[gt_boundary > 0] <= tolerance)) / gt_count
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def relative_area_error(prediction, ground_truth):
    """Absolute per-fiber area error, divided by the ground-truth area."""
    ground_truth_area = float(cv2.contourArea(ground_truth))
    if ground_truth_area <= 1e-06:
        return float('inf')
    return abs(float(cv2.contourArea(prediction)) - ground_truth_area) / ground_truth_area

def iter_class_contours(grouped, class_names):
    instances = []
    for class_index, class_name in enumerate(class_names):
        for contour in grouped.get(class_name, []):
            instances.append((class_index, contour))
    return instances

def match_instances(predictions, ground_truth, score_fn, *, higher_is_better, threshold, tolerance):
    """Greedy one-to-one matches. ``score_fn`` is boundary F1 or area error."""
    pairs = []
    for pred_index, (_, pred_contour) in enumerate(predictions):
        for gt_index, (_, gt_contour) in enumerate(ground_truth):
            if not contours_correspond(pred_contour, gt_contour, tolerance):
                continue
            score = score_fn(pred_contour, gt_contour)
            if higher_is_better and score < threshold:
                continue
            if not higher_is_better and score > threshold:
                continue
            pairs.append((score, centroid_distance(pred_contour, gt_contour), pred_index, gt_index))
    if higher_is_better:
        pairs.sort(key=lambda item: (-item[0], item[1]))
    else:
        pairs.sort(key=lambda item: (item[0], item[1]))
    used_predictions = set()
    used_ground_truth = set()
    matches = []
    for score, _, pred_index, gt_index in pairs:
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index, score))
    return matches

def apply_matches(matrix, matches, predictions, ground_truth, scores_by_class):
    matched_predictions = set()
    matched_ground_truth = set()
    background = matrix.shape[0] - 1
    for pred_index, gt_index, score in matches:
        pred_class = predictions[pred_index][0]
        gt_class = ground_truth[gt_index][0]
        matrix[gt_class, pred_class] += 1
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        if pred_class == gt_class:
            scores_by_class[pred_class].append(score)
    for pred_index, (pred_class, _) in enumerate(predictions):
        if pred_index not in matched_predictions:
            matrix[background, pred_class] += 1
    for gt_index, (gt_class, _) in enumerate(ground_truth):
        if gt_index not in matched_ground_truth:
            matrix[gt_class, background] += 1

def accumulate_confusion_matrices(boundary_matrix, area_matrix, boundary_scores, area_scores, *, predictions, label_path, image_shape, class_names, edge_margin, min_cotton_area, min_lumen_area, max_area_ratio, boundary_f1_threshold, boundary_tolerance, area_error_threshold):
    """Score one labeled image after edge filtering. Returns skipped ground-truth edges."""
    image_height, image_width = image_shape[:2]
    ground_truth = read_yolo_contours(label_path, image_width, image_height, class_names)
    filtered_predictions = {}
    filtered_ground_truth = {}
    skipped_edge = 0
    for class_name in class_names:
        # Predictions are the contours analysis kept: window-edge rejection, dedupe,
        # and the large-contour filter already ran. Drop anything still on the image border.
        filtered_predictions[class_name] = []
        for contour in predictions.get(class_name, []):
            if contour_touches_window_or_image_edge(contour, valid_width=image_width, valid_height=image_height, margin=edge_margin):
                continue
            filtered_predictions[class_name].append(contour)
        filtered_ground_truth[class_name], class_skipped = filter_boundary_contours(ground_truth.get(class_name, []), class_name=class_name, image_shape=image_shape, edge_margin=edge_margin, min_cotton_area=min_cotton_area, min_lumen_area=min_lumen_area, max_area_ratio=max_area_ratio)
        skipped_edge += class_skipped
    prediction_instances = iter_class_contours(filtered_predictions, class_names)
    ground_truth_instances = iter_class_contours(filtered_ground_truth, class_names)
    boundary_matches = match_instances(prediction_instances, ground_truth_instances, lambda pred, gt: boundary_f1(pred, gt, boundary_tolerance), higher_is_better=True, threshold=boundary_f1_threshold, tolerance=boundary_tolerance)
    area_matches = match_instances(prediction_instances, ground_truth_instances, relative_area_error, higher_is_better=False, threshold=area_error_threshold, tolerance=boundary_tolerance)
    apply_matches(boundary_matrix, boundary_matches, prediction_instances, ground_truth_instances, boundary_scores)
    apply_matches(area_matrix, area_matches, prediction_instances, ground_truth_instances, area_scores)
    return skipped_edge

def class_detection_metrics(matrix, class_index, scores):
    true_positive = int(matrix[class_index, class_index])
    false_positive = int(matrix[:, class_index].sum() - true_positive)
    false_negative = int(matrix[class_index, :].sum() - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_score = float(np.mean(scores)) if scores else None
    return (true_positive, false_positive, false_negative, precision, recall, f1, mean_score)

def write_matrix_csv(path, matrix, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['true\\predicted', *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *(int(value) for value in row)])

def normalize_confusion_matrix(matrix):
    """Row-normalize a count matrix so each true class sums to 1 when that row has counts."""
    counts = np.asarray(matrix, dtype=np.float64)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    return normalized

def plot_confusion_matrix(path, matrix, class_names, title, subtitle):
    """Save a row-normalized confusion matrix. Cell color follows the row fraction."""
    import sys
    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(class_names)
    counts = np.asarray(matrix)
    normalized = normalize_confusion_matrix(counts)
    figure, axis = plt.subplots(figsize=(6.8, 5.6))
    image = axis.imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha='right')
    axis.set_yticklabels(labels)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    axis.set_title(title)
    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            fraction = float(normalized[row_index, column_index])
            count = int(counts[row_index, column_index])
            color = 'white' if fraction > 0.55 else 'black'
            axis.text(
                column_index,
                row_index,
                f'{fraction:.2f}\n({count})',
                ha='center',
                va='center',
                color=color,
                fontsize=10,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label('row-normalized fraction')
    figure.text(0.5, 0.01, subtitle, ha='center', va='bottom', fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

def write_confusion_outputs(output_dir, *, class_names, boundary_matrix, area_matrix, boundary_scores, area_scores, labeled_images, edge_margin, boundary_f1_threshold, boundary_tolerance, area_error_threshold):
    """Write both confusion matrices, their plots, and per-class match scores."""
    labels = list(class_names) + ['background']
    matrix_dir = output_dir / 'confusion_matrices'
    image_word = 'image' if labeled_images == 1 else 'images'
    subtitle = f'{labeled_images} labeled {image_word}, edge margin {edge_margin} px, boundary detections excluded'
    boundary_title = f'Confusion matrix — boundary F1 ≥ {boundary_f1_threshold:.2f} ({boundary_tolerance:g} px)'
    area_title = f'Confusion matrix — area error ≤ {area_error_threshold:.0%}'
    write_matrix_csv(matrix_dir / 'boundary_f1.csv', boundary_matrix, labels)
    write_matrix_csv(matrix_dir / 'area_error.csv', area_matrix, labels)
    plot_confusion_matrix(matrix_dir / 'boundary_f1.png', boundary_matrix, labels, boundary_title, subtitle)
    plot_confusion_matrix(matrix_dir / 'area_error.png', area_matrix, labels, area_title, subtitle)
    metrics_path = matrix_dir / 'metrics.csv'
    with metrics_path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['metric', 'class', 'threshold', 'tolerance_px', 'edge_margin_px', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_match_score'])
        metric_rows = (
            ('boundary_f1', boundary_matrix, boundary_scores, boundary_f1_threshold, boundary_tolerance),
            ('area_error', area_matrix, area_scores, area_error_threshold, ''),
        )
        for metric_name, matrix, scores, threshold, tolerance in metric_rows:
            for class_index, class_name in enumerate(class_names):
                true_positive, false_positive, false_negative, precision, recall, f1, mean_score = class_detection_metrics(matrix, class_index, scores[class_index])
                writer.writerow([
                    metric_name,
                    class_name,
                    f'{threshold:.6f}',
                    tolerance if tolerance == '' else f'{tolerance:.6f}',
                    edge_margin,
                    true_positive,
                    false_positive,
                    false_negative,
                    f'{precision:.6f}',
                    f'{recall:.6f}',
                    f'{f1:.6f}',
                    '' if mean_score is None else f'{mean_score:.6f}',
                ])
                print(f'  {metric_name} {class_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (tp={true_positive} fp={false_positive} fn={false_negative})')
    print(f'Wrote confusion matrices to {matrix_dir}')

def run_analysis(args):
    started = time.perf_counter()
    _require_yolo_dependencies()
    if args.um_per_px is not None and args.um_per_px <= 0:
        raise ValueError('--um-per-px must be greater than 0')
    if args.imgsz <= 0:
        raise ValueError('--imgsz must be greater than 0')
    if args.window_size <= 0:
        raise ValueError('--window-size must be greater than 0')
    if args.window_stride <= 0:
        raise ValueError('--window-stride must be greater than 0')
    if args.window_edge_margin < 0:
        raise ValueError('--window-edge-margin must be greater than or equal to 0')
    if args.dedupe_iou < 0 or args.dedupe_iou > 1:
        raise ValueError('--dedupe-iou must be between 0 and 1')
    if args.large_contour_stddevs < 0:
        raise ValueError('--large-contour-stddevs must be greater than or equal to 0')
    if args.large_contour_min_samples < 1:
        raise ValueError('--large-contour-min-samples must be greater than or equal to 1')
    if args.boundary_f1_threshold < 0 or args.boundary_f1_threshold > 1:
        raise ValueError('--boundary-f1-threshold must be between 0 and 1')
    if args.boundary_tolerance < 0:
        raise ValueError('--boundary-tolerance must be greater than or equal to 0')
    if args.area_error_threshold < 0:
        raise ValueError('--area-error-threshold must be greater than or equal to 0')
    if args.labels_dir is not None and not args.labels_dir.is_dir():
        raise FileNotFoundError(f'Label directory not found: {args.labels_dir}')
    model = YOLO(str(args.model))
    extensions = parse_extensions(args.extensions)
    images = iter_images(args.image_dir, extensions, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found in {args.image_dir} with extensions {', '.join(extensions)}")
    output_csv = args.output_csv or args.output_dir / 'fiber_measurements.csv'
    class_names = tuple(name.strip().lower() for name in args.class_names)
    stats = MeasurementTiming(class_names)
    all_rows = []
    object_index = 0
    labeled_images = 0
    skipped_edge_ground_truth = 0
    boundary_matrix = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=np.int64)
    area_matrix = np.zeros_like(boundary_matrix)
    boundary_scores = [[] for _ in class_names]
    area_scores = [[] for _ in class_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_number, image_path in enumerate(images, start=1):
        print(f'[{image_number}/{len(images)}] {image_path}')
        image_started = time.perf_counter()
        before_image = stats.snapshot()
        read_started = time.perf_counter()
        image = read_image_bgr(image_path)
        stats.image_read_seconds += time.perf_counter() - read_started
        stats.images_processed += 1
        predict_options = {'show_labels': False, 'line_width': args.line_width, 'conf': args.confidence, 'iou': args.iou, 'verbose': args.verbose, 'imgsz': args.imgsz, 'retina_masks': True}
        if args.device is not None:
            predict_options['device'] = args.device
        rows, object_index, cotton_contours, lumen_contours, conversion_factor = analyze_image_sliding_window(model, image=image, image_path=image_path, output_dir=args.output_dir, class_names=class_names, predict_options=predict_options, window_size=args.window_size, window_stride=args.window_stride, window_edge_margin=args.window_edge_margin, dedupe_iou=args.dedupe_iou, large_contour_stddevs=args.large_contour_stddevs, large_contour_min_samples=args.large_contour_min_samples, min_cotton_area=args.min_cotton_area, min_lumen_area=args.min_lumen_area, max_area_ratio=args.max_area_ratio, overlay_alpha=args.overlay_alpha, save_overlays=args.save_overlays, verbose=args.verbose, start_index=object_index, um_per_px=args.um_per_px, stats=stats, full_metrics=args.full_metrics)
        all_rows.extend(rows)
        label_path = label_path_for_image(image_path, args.image_dir, args.labels_dir)
        if label_path is not None and label_path.is_file():
            labeled_images += 1
            confusion_started = time.perf_counter()
            skipped_edge_ground_truth += accumulate_confusion_matrices(boundary_matrix, area_matrix, boundary_scores, area_scores, predictions={'cotton': cotton_contours, 'lumen': lumen_contours}, label_path=label_path, image_shape=image.shape, class_names=class_names, edge_margin=args.window_edge_margin, min_cotton_area=args.min_cotton_area, min_lumen_area=args.min_lumen_area, max_area_ratio=args.max_area_ratio, boundary_f1_threshold=args.boundary_f1_threshold, boundary_tolerance=args.boundary_tolerance, area_error_threshold=args.area_error_threshold)
            stats.confusion_seconds += time.perf_counter() - confusion_started
        print_measurement_summary(stats.delta_since(before_image), time.perf_counter() - image_started, um_per_px=conversion_factor, heading=f'Image summary: {image_path.name}')
    write_started = time.perf_counter()
    write_rows(output_csv, all_rows)
    stats.write_seconds += time.perf_counter() - write_started
    print(f'Wrote {len(all_rows)} rows for {object_index} cotton objects to {output_csv}')
    if labeled_images:
        print(f'Confusion matrices from {labeled_images} labeled images ({skipped_edge_ground_truth} edge ground-truth instances excluded):')
        write_confusion_outputs(args.output_dir, class_names=class_names, boundary_matrix=boundary_matrix, area_matrix=area_matrix, boundary_scores=boundary_scores, area_scores=area_scores, labeled_images=labeled_images, edge_margin=args.window_edge_margin, boundary_f1_threshold=args.boundary_f1_threshold, boundary_tolerance=args.boundary_tolerance, area_error_threshold=args.area_error_threshold)
    print_measurement_summary(stats, time.perf_counter() - started)
    return 0

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='Path to YOLO segmentation weights.')
    parser.add_argument('--um-per-px', type=float, help='Micrometers per pixel applied to every image. When omitted, the yellow scale bar is measured and a missing bar is an error.')
    parser.add_argument('--image-dir', required=True, type=Path, help='Directory containing images to analyze.')
    parser.add_argument('--output-dir', type=Path, default=Path('fiber_output'), help='Directory for analysis outputs.')
    parser.add_argument('--output-csv', type=Path, help='Aggregate CSV path. Defaults to output-dir/fiber_measurements.csv.')
    parser.add_argument('--extensions', nargs='+', default=DEFAULT_EXTENSIONS, help='Image file extensions to include.')
    parser.add_argument('--recursive', action='store_true', help='Search image-dir recursively.')
    parser.add_argument('--class-names', nargs='+', default=DEFAULT_CLASS_NAMES, help='Fallback class names by YOLO index.')
    parser.add_argument('--min-cotton-area', type=float, default=100.0, help='Minimum cotton contour area in pixels.')
    parser.add_argument('--min-lumen-area', type=float, default=50.0, help='Minimum lumen contour area in pixels.')
    parser.add_argument('--max-area-ratio', type=float, default=0.95, help='Discard masks larger than this fraction of window area.')
    parser.add_argument('--imgsz', type=int, default=960, help='YOLO inference size. Each window is letterboxed to this size; masks are mapped back to the window.')
    parser.add_argument('--window-size', type=int, default=512, help='Sliding-window square size for inference.')
    parser.add_argument('--window-stride', type=int, default=256, help='Sliding-window stride in pixels.')
    parser.add_argument('--window-edge-margin', type=int, default=2, help='Ignore detections this close to a window seam or the composite image border. Confusion matrices also drop ground truth on the image border by this margin.')
    parser.add_argument('--labels-dir', type=Path, help='YOLO polygon labels aligned with --image-dir. Defaults to replacing "images" with "labels" in each image path. Confusion matrices are written when a label file exists.')
    parser.add_argument('--boundary-f1-threshold', type=float, default=0.5, help='Minimum boundary F1 for a true positive in the boundary-F1 confusion matrix.')
    parser.add_argument('--boundary-tolerance', type=float, default=2.0, help='Pixel distance that counts as a boundary hit for boundary F1.')
    parser.add_argument('--area-error-threshold', type=float, default=0.25, help='Maximum relative per-fiber area error |pred-gt|/gt for a true positive in the area-error confusion matrix. This scores lumen without mask IoU, which collapses when a small lumen shifts.')
    parser.add_argument('--dedupe-iou', type=float, default=0.5, help='Remove duplicate same-class detections with bounding-box IoU at or above this value.')
    parser.add_argument('--large-contour-stddevs', type=float, default=3.0, help='Remove contours larger than mean area plus this many standard deviations. Use 0 to disable.')
    parser.add_argument('--large-contour-min-samples', type=int, default=6, help='Minimum contour count before applying the large-contour outlier filter.')
    parser.add_argument('--confidence', type=float, default=0.25, help='YOLO confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.7, help='YOLO IoU threshold.')
    parser.add_argument('--device', help='YOLO device, e.g. cpu, 0, or cuda:0.')
    parser.add_argument('--line-width', type=int, default=1, help='Line width for YOLO-rendered results.')
    parser.add_argument('--save-overlays', action='store_true', default=True, help='Save one merged cotton/lumen overlay for each source image.')
    parser.add_argument('--no-save-overlays', action='store_false', dest='save_overlays', help='Disable merged overlay image output.')
    parser.add_argument('--overlay-alpha', type=float, default=0.4, help='Overlay opacity.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose YOLO logging.')
    parser.add_argument('--full-metrics', action='store_true', help='Also compute morphology beyond area and perimeter for each cotton and lumen. Those metrics are skipped unless this is set.')
    return parser

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return run_analysis(args)
if __name__ == '__main__':
    raise SystemExit(main())
