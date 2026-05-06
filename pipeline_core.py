import os
import math
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = os.path.dirname(__file__)

# Modelo YOLO da folha A4
A4_MODEL_DIR  = "models"
A4_MODEL_PATH = os.path.join(A4_MODEL_DIR, "a4_best.pt")   # <<< ajuste conforme seu treino

# Modelo YOLO da ROI / braço (PPD)
PPD_MODEL_DIR  = "models"
PPD_MODEL_PATH = os.path.join(PPD_MODEL_DIR, "ppd_best.pt")  # <<< já usávamos esse

CONF_THRES_A4 = 0.25
CONF_THRES_PPD = 0.25
YOLO_DEVICE = "cpu"

A4_W_MM = 210.0
A4_H_MM = 297.0

# ================= LOAD MODELS =================
#a4_model = YOLO(A4_MODEL_PATH)
#ppd_model = YOLO(PPD_MODEL_PATH)
a4_model = None
ppd_model = None

def load_models():
    global a4_model, ppd_model

    if a4_model is None:
        print("Carregando modelo A4...")
        a4_model = YOLO(A4_MODEL_PATH)

    if ppd_model is None:
        print("Carregando modelo PPD...")
        ppd_model = YOLO(PPD_MODEL_PATH)

# ================= HELPERS (IGUAIS) =================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def bbox_cxcywh_to_xyxy(cx, cy, bw, bh, w_img, h_img, normalized=True, margin_frac=0.0):
    if normalized:
        cx *= w_img; cy *= h_img; bw *= w_img; bh *= h_img

    x1 = cx - bw/2
    y1 = cy - bh/2
    x2 = cx + bw/2
    y2 = cy + bh/2

    mx = margin_frac * bw
    my = margin_frac * bh

    x1 -= mx; x2 += mx
    y1 -= my; y2 += my

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w_img, int(round(x2)))
    y2 = min(h_img, int(round(y2)))

    return x1, y1, x2, y2

def largest_component(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out

def fill_holes(mask):
    inv = cv2.bitwise_not(mask)
    h,w = mask.shape[:2]
    flood = inv.copy()
    mask_flood = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask_flood, (0,0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return mask | flood_inv

# ================= SEGMENTAÇÃO (IDÊNTICA) =================

def segment_roi(roi, out_dir):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    mask_skin = cv2.medianBlur(mask_skin, 7)
    mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    mask_skin = largest_component(mask_skin)

    cv2.imwrite(os.path.join(out_dir, "roi_mask_skin.jpg"), mask_skin)

    # ================= HISTOGRAMA =================
    hue_vals = hsv[:,:,0][mask_skin > 0]
    hist = cv2.calcHist([hue_vals], [0], None, [180], [0,180]).ravel().astype(float)
    hist_s = gaussian_filter1d(hist, sigma=2.0)

    peaks, _ = find_peaks(hist_s, prominence=max(hist_s.max()*0.02, 3.0), distance=3)

    if len(peaks) > 0:
        main_peak = int(peaks[np.argmax(hist_s[peaks])])
        std_h = float(np.std(hue_vals))
        start_peak = int(main_peak - std_h)
        end_peak   = int(main_peak + 1.5 * std_h)
    else:
        mean_h = float(np.mean(hue_vals))
        std_h = float(np.std(hue_vals))
        start_peak = int(mean_h - 1.5 * std_h)
        end_peak   = int(mean_h + 1.5 * std_h)

    start_peak = max(0, start_peak)
    end_peak = min(179, end_peak)

    lower = np.array([start_peak, 20, 50], np.uint8)
    upper = np.array([end_peak, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, mask, mask=mask_skin)

    cv2.imwrite(os.path.join(out_dir, "roi_mask_final.jpg"), mask)

    return mask

# ================= MAIN PIPELINE =================

def process_image(image, output_dir):
    load_models()
    ensure_dir(output_dir)

    H, W = image.shape[:2]

    # ================= A4 =================
    a4_res = a4_model(image, conf=CONF_THRES_A4, device=YOLO_DEVICE)[0]

    mm2_per_px2 = None

    if a4_res.boxes is not None and len(a4_res.boxes) > 0:
        box = a4_res.boxes[0].xywhn[0].cpu().numpy()
        x1, y1, x2, y2 = bbox_cxcywh_to_xyxy(*box, W, H)

        w_px = x2 - x1
        h_px = y2 - y1

        mm2_per_px2 = (A4_W_MM / w_px) * (A4_H_MM / h_px)

    # ================= PPD =================
    ppd_res = ppd_model(image, conf=CONF_THRES_PPD, device=YOLO_DEVICE)[0]

    if ppd_res.boxes is None or len(ppd_res.boxes) == 0:
        return None

    box = ppd_res.boxes[0].xywhn[0].cpu().numpy()
    x1, y1, x2, y2 = bbox_cxcywh_to_xyxy(*box, W, H, margin_frac=0.10)

    roi = image[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_dir, "roi.jpg"), roi)

    # ================= SEGMENTAÇÃO =================
    mask = segment_roi(roi, output_dir)

    area_px = int(np.count_nonzero(mask))

    # ================= FULL MASK =================
    mask_full = np.zeros((H, W), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask

    cv2.imwrite(os.path.join(output_dir, "mask_full.png"), mask_full)

    overlay = image.copy()
    overlay[mask_full > 0] = (0, 0, 255)

    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imwrite(os.path.join(output_dir, "overlay.png"), result)

    # ================= MEDIDAS =================
    area_mm2 = None
    radius_mm = None

    if mm2_per_px2:
        area_mm2 = area_px * mm2_per_px2
        radius_mm = math.sqrt(area_mm2 / math.pi)

    return {
        "area_px": area_px,
        "area_mm2": area_mm2,
        "radius_mm": radius_mm
    }