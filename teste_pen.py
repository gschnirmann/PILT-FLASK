import os
import cv2
import numpy as np

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


# =========================================
# UTIL
# =========================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================
# A4 VIA YOLO
# =========================================
def detect_a4_with_yolo(image_bgr, yolo_model):
    results = yolo_model(image_bgr, verbose=False)

    if not results or len(results) == 0:
        return None, None, None

    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return None, None, None

    best_box = None
    best_conf = -1.0

    for box in result.boxes:
        conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return None, None, None

    xyxy = best_box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy)

    a4_box = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    w_px = max(1.0, float(x2 - x1))
    h_px = max(1.0, float(y2 - y1))

    long_px = max(w_px, h_px)
    short_px = min(w_px, h_px)

    px_per_mm_long = long_px / A4_HEIGHT_MM
    px_per_mm_short = short_px / A4_WIDTH_MM
    px_per_mm = (px_per_mm_long + px_per_mm_short) / 2.0

    a4_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(a4_mask, [a4_box], -1, 255, thickness=-1)

    return a4_box, px_per_mm, a4_mask


# =========================================
# DETECÇÃO ROBUSTA DO AZUL
# =========================================
def detect_blue_pen_mask(image_bgr):
    """
    Detecção azul mais robusta:
    - HSV amplo
    - Lab
    - dominância do canal B
    - saturação mínima
    - reforço adaptativo
    """
    b, g, r = cv2.split(image_bgr)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_, a_, b_lab = cv2.split(lab)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([90, 35, 20], dtype=np.uint8),
        np.array([145, 255, 255], dtype=np.uint8)
    )

    lab_mask = cv2.inRange(b_lab, 0, 145)

    blue_dom_1 = cv2.subtract(b, g)
    blue_dom_2 = cv2.subtract(b, r)

    _, dom_mask_1 = cv2.threshold(blue_dom_1, 12, 255, cv2.THRESH_BINARY)
    _, dom_mask_2 = cv2.threshold(blue_dom_2, 12, 255, cv2.THRESH_BINARY)

    dom_mask = cv2.bitwise_and(dom_mask_1, dom_mask_2)

    _, sat_mask = cv2.threshold(s, 28, 255, cv2.THRESH_BINARY)

    base_mask = cv2.bitwise_and(hsv_mask, lab_mask)
    base_mask = cv2.bitwise_and(base_mask, dom_mask)
    base_mask = cv2.bitwise_and(base_mask, sat_mask)

    score = (
        1.2 * blue_dom_1.astype(np.float32) +
        1.2 * blue_dom_2.astype(np.float32) +
        0.6 * s.astype(np.float32) -
        0.3 * b_lab.astype(np.float32)
    )

    plausible = score[base_mask > 0]
    adaptive_mask = np.zeros_like(base_mask)

    if plausible.size > 20:
        thr = np.percentile(plausible, 25)
        adaptive_mask[score >= thr] = 255
        adaptive_mask = cv2.bitwise_and(adaptive_mask, hsv_mask)
        adaptive_mask = cv2.bitwise_and(adaptive_mask, sat_mask)

    final_mask = cv2.bitwise_or(base_mask, adaptive_mask)

    return final_mask


def refine_blue_mask(mask):
    """
    Preserva traço fino e conecta microfalhas.
    """
    kernel2 = np.ones((2, 2), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)

    refined = cv2.dilate(mask, kernel2, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel3, iterations=1)

    return refined


def keep_relevant_blue_fragments(mask, min_component_area=4):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == i] = 255

    return filtered


# =========================================
# RECONSTRUÇÃO CIRCULAR SUAVE
# =========================================
def estimate_center_from_fragments(fragment_mask):
    ys, xs = np.where(fragment_mask > 0)
    if len(xs) < 5:
        return None

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return cx, cy


def moving_average_circular(arr, window=9):
    if window < 3:
        return arr.copy()

    pad = window // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(ext, kernel, mode="valid")
    return smoothed


def reconstruct_circular_contour(fragment_mask, angle_step_deg=2, smooth_window=11):
    """
    Reconstrói um contorno fechado suave com tendência circular,
    interpolando os raios ausentes por ângulo.
    """
    ys, xs = np.where(fragment_mask > 0)
    if len(xs) < 10:
        return np.zeros_like(fragment_mask), None, None

    center = estimate_center_from_fragments(fragment_mask)
    if center is None:
        return np.zeros_like(fragment_mask), None, None

    cx, cy = center

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    angles = np.degrees(np.arctan2(dy, dx))
    angles = (angles + 360.0) % 360.0
    radii = np.sqrt(dx * dx + dy * dy)

    bins = np.arange(0, 360, angle_step_deg, dtype=np.float32)
    radius_by_bin = np.full(len(bins), np.nan, dtype=np.float32)

    half_bin = angle_step_deg / 2.0

    for i, a in enumerate(bins):
        diff = np.abs(((angles - a + 180) % 360) - 180)
        idx = np.where(diff <= half_bin)[0]
        if len(idx) > 0:
            radius_by_bin[i] = np.median(radii[idx])

    valid = ~np.isnan(radius_by_bin)
    if np.sum(valid) < max(8, len(bins) * 0.15):
        return np.zeros_like(fragment_mask), None, None

    valid_idx = np.where(valid)[0]
    valid_r = radius_by_bin[valid]

    xp = valid_idx.astype(np.float32)
    fp = valid_r.astype(np.float32)
    xfull = np.arange(len(bins), dtype=np.float32)

    xp_ext = np.concatenate([xp - len(bins), xp, xp + len(bins)])
    fp_ext = np.concatenate([fp, fp, fp])

    interp_r = np.interp(xfull, xp_ext, fp_ext)
    smooth_r = moving_average_circular(interp_r, window=smooth_window)

    pts = []
    for i, a in enumerate(bins):
        theta = np.radians(a)
        r = float(smooth_r[i])
        x = int(round(cx + r * np.cos(theta)))
        y = int(round(cy + r * np.sin(theta)))
        pts.append([x, y])

    contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    filled_mask = np.zeros_like(fragment_mask)
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

    return filled_mask, contour, (cx, cy)


# =========================================
# MEDIÇÃO
# =========================================
def measure_mask(mask, px_per_mm):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "area_px": 0,
            "area_mm2": 0.0,
            "perimeter_px": 0.0,
            "perimeter_mm": 0.0,
            "major_axis_px": 0.0,
            "major_axis_mm": 0.0,
            "minor_axis_px": 0.0,
            "minor_axis_mm": 0.0,
            "equivalent_diameter_px": 0.0,
            "equivalent_diameter_mm": 0.0,
            "bbox": None,
        }

    cnt = max(contours, key=cv2.contourArea)

    area_px = int(cv2.contourArea(cnt))
    perimeter_px = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)

    major_axis_px = float(max(w, h))
    minor_axis_px = float(min(w, h))
    equivalent_diameter_px = float(np.sqrt((4.0 * area_px) / np.pi)) if area_px > 0 else 0.0

    if px_per_mm is not None and px_per_mm > 0:
        mm_per_px = 1.0 / px_per_mm
        area_mm2 = area_px * (mm_per_px ** 2)
        perimeter_mm = perimeter_px * mm_per_px
        major_axis_mm = major_axis_px * mm_per_px
        minor_axis_mm = minor_axis_px * mm_per_px
        equivalent_diameter_mm = equivalent_diameter_px * mm_per_px
    else:
        area_mm2 = 0.0
        perimeter_mm = 0.0
        major_axis_mm = 0.0
        minor_axis_mm = 0.0
        equivalent_diameter_mm = 0.0

    return {
        "area_px": area_px,
        "area_mm2": area_mm2,
        "perimeter_px": perimeter_px,
        "perimeter_mm": perimeter_mm,
        "major_axis_px": major_axis_px,
        "major_axis_mm": major_axis_mm,
        "minor_axis_px": minor_axis_px,
        "minor_axis_mm": minor_axis_mm,
        "equivalent_diameter_px": equivalent_diameter_px,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "bbox": (x, y, w, h),
    }


# =========================================
# OVERLAY
# =========================================
def build_overlay(image_bgr, a4_box, fragment_mask, circular_contour, enduration_mask, metrics, center):
    overlay = image_bgr.copy()

    if a4_box is not None:
        cv2.drawContours(overlay, [a4_box], -1, (255, 255, 0), 2)

    if fragment_mask is not None and np.sum(fragment_mask > 0) > 0:
        frag_contours, _ = cv2.findContours(fragment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, frag_contours, -1, (0, 255, 255), 1)

    if circular_contour is not None:
        cv2.drawContours(overlay, [circular_contour], -1, (255, 0, 255), 2)

    if center is not None:
        cx, cy = center
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), -1)

    if enduration_mask is not None and np.sum(enduration_mask > 0) > 0:
        color_layer = np.zeros_like(overlay)
        color_layer[:, :] = (0, 255, 0)
        colored = cv2.bitwise_and(color_layer, color_layer, mask=enduration_mask)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.25, 0)

        filled_contours, _ = cv2.findContours(enduration_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, filled_contours, -1, (0, 200, 0), 2)

    if metrics["bbox"] is not None:
        x, y, w, h = metrics["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    y0 = 30
    step = 28
    lines = [
        f"area_px: {metrics['area_px']}",
        f"area_mm2: {metrics['area_mm2']:.2f}",
        f"perimeter_mm: {metrics['perimeter_mm']:.2f}",
        f"eq_diam_mm: {metrics['equivalent_diameter_mm']:.2f}",
    ]

    for line in lines:
        cv2.putText(
            overlay,
            line,
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y0 += step

    return overlay


# =========================================
# PIPELINE
# =========================================
def run_pipeline(image_path, yolo_model, output_dir="test_outputs"):
    ensure_dir(output_dir)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    sample_dir = os.path.join(output_dir, base_name)
    ensure_dir(sample_dir)

    a4_box, px_per_mm, _ = detect_a4_with_yolo(image, yolo_model)

    blue_mask = detect_blue_pen_mask(image)
    refined_blue_mask = refine_blue_mask(blue_mask)
    fragment_mask = keep_relevant_blue_fragments(refined_blue_mask, min_component_area=4)

    enduration_mask, circular_contour, center = reconstruct_circular_contour(
        fragment_mask,
        angle_step_deg=2,
        smooth_window=11,
    )

    metrics = measure_mask(enduration_mask, px_per_mm)

    overlay = build_overlay(
        image,
        a4_box,
        fragment_mask,
        circular_contour,
        enduration_mask,
        metrics,
        center,
    )

    a4_vis = image.copy()
    if a4_box is not None:
        cv2.drawContours(a4_vis, [a4_box], -1, (255, 255, 0), 2)

    cv2.imwrite(os.path.join(sample_dir, "01_a4_detected.png"), a4_vis)
    cv2.imwrite(os.path.join(sample_dir, "02_blue_mask_raw.png"), blue_mask)
    cv2.imwrite(os.path.join(sample_dir, "03_blue_mask_refined.png"), refined_blue_mask)
    cv2.imwrite(os.path.join(sample_dir, "04_fragment_mask.png"), fragment_mask)
    cv2.imwrite(os.path.join(sample_dir, "05_enduration_mask.png"), enduration_mask)
    cv2.imwrite(os.path.join(sample_dir, "06_overlay.png"), overlay)

    return {
        "input_path": image_path,
        "output_dir": sample_dir,
        "px_per_mm": float(px_per_mm) if px_per_mm is not None else None,
        "area_px": metrics["area_px"],
        "area_mm2": metrics["area_mm2"],
        "perimeter_px": metrics["perimeter_px"],
        "perimeter_mm": metrics["perimeter_mm"],
        "major_axis_px": metrics["major_axis_px"],
        "major_axis_mm": metrics["major_axis_mm"],
        "minor_axis_px": metrics["minor_axis_px"],
        "minor_axis_mm": metrics["minor_axis_mm"],
        "equivalent_diameter_px": metrics["equivalent_diameter_px"],
        "equivalent_diameter_mm": metrics["equivalent_diameter_mm"],
        "bbox": metrics["bbox"],
    }


# =========================================
# EXEMPLO DE USO
# =========================================
if __name__ == "__main__":
    from ultralytics import YOLO

    image_path = "data_pen/teste5.png"
    model_path = "/home/guilherme/Documents/PILT/Projeto-PILT-Flask/models/a4_best.pt"

    yolo_model = YOLO(model_path)

    result = run_pipeline(
        image_path=image_path,
        yolo_model=yolo_model,
        output_dir="test_outputs"
    )

    print("\n=== RESULTADO FINAL ===")
    for k, v in result.items():
        print(f"{k}: {v}")