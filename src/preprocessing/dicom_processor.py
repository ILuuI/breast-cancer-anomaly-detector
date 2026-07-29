"""
Pixel Monotonic Processor: pipeline de estandarizacion radiologica de
imagenes DICOM (Modality LUT, VOI LUT, segmentacion anatomica,
winsorizacion robusta y mapeo de tono monotonico) previo a la
cuantizacion final a 8 bits.

Tambien incluye GeometryManager, encargado de la estandarizacion de
lateralidad y el recorte automatico de la region de interes (ROI).
"""
import cv2
import numpy as np

try:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

from skimage.measure import label, regionprops
from skimage.morphology import closing, disk

from ui.theme import norm8

class PixelMonotonicProcessor:
    def __init__(self, gamma=1.8, alpha=0.05, k=10.0):
        self.gamma, self.alpha, self.k = gamma, alpha, k

    def extract_breast_mask(self, image):
        blurred = cv2.GaussianBlur(image.astype(np.float32), (19, 19), 0)
        p01, p99 = float(np.percentile(blurred, 1)), float(np.percentile(blurred, 99))
        norm   = np.clip((blurred - p01) / (p99 - p01 + 1e-8), 0, 1)
        binary = (norm > 0.02).astype(np.uint8)
        labeled = label(binary)
        regions = regionprops(labeled)
        if not regions: return np.zeros_like(image, dtype=bool)
        largest = max(regions, key=lambda r: r.area)
        if largest.area < 0.02 * image.size: return np.zeros_like(image, dtype=bool)
        return closing(labeled == largest.label, disk(15))

    def process_steps(self, dcm):
        steps = []
        try:   after_mod = apply_modality_lut(dcm.pixel_array, dcm).astype(np.float32)
        except Exception: after_mod = dcm.pixel_array.astype(np.float32)
        steps.append(("Lectura de Array DICOM", norm8(after_mod)))

        try:   raw = apply_voi_lut(after_mod, dcm).astype(np.float32)
        except Exception: raw = after_mod
        
        photo = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
        if photo == "MONOCHROME1": 
            raw = float(raw.max()) - raw + float(raw.min())
            
        steps.append(("Correccion Fotometrica LUT", norm8(raw)))
        blurred = cv2.GaussianBlur(raw.astype(np.float32), (19, 19), 0)
        steps.append(("Filtro Gaussiano Anatomico", norm8(blurred)))

        p01, p99 = float(np.percentile(blurred, 1)), float(np.percentile(blurred, 99))
        nb  = np.clip((blurred - p01) / (p99 - p01 + 1e-8), 0, 1)
        binary = (nb > 0.02).astype(np.uint8)
        steps.append(("Umbralizacion Dinamica", (binary * 255).astype(np.uint8)))

        labeled = label(binary)
        regions = regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            mask    = closing(labeled == largest.label, disk(15))
        else: mask = np.zeros_like(raw, dtype=bool)
        mv = np.zeros_like(raw, dtype=np.uint8); mv[mask] = 255
        steps.append(("Operador Morfologico (Closing)", mv))

        image = raw.copy()
        if mask.sum() > 0:
            tissue = image[mask]
            q_lo, q_hi = float(np.percentile(tissue, 0.1)), float(np.percentile(tissue, 99.9))
            norm_  = np.clip((image - q_lo) / (q_hi - q_lo + 1e-6), 0, 1)
        else:
            norm_  = np.clip((image - image.min()) / (image.max() - image.min() + 1e-6), 0, 1)
        norm_[~mask] = 0
        steps.append(("Normalizacion de Tejido (p1-p99)", np.clip(norm_ * 255, 0, 255).astype(np.uint8)))

        t_dyn = float(np.median(norm_[mask])) if mask.sum() > 0 else 0.5
        g, a, k = 1.8, 0.05, 10.0
        mono = ((1 - a) * np.power(np.clip(norm_, 0, 1), g) + a * (1.0 / (1.0 + np.exp(-np.clip(k * (norm_ - t_dyn), -10, 10)))))
        final_8b = np.clip(mono * 255, 0, 255).astype(np.uint8)
        if mask.any():
            bg = int(final_8b[mask].min())
            if bg < 30: final_8b = np.clip(final_8b.astype(np.int16) - bg, 0, 255).astype(np.uint8)
        final_8b[~mask] = 0
        steps.append(("Transformada Monotonica Hibrida", final_8b.copy()))
        return steps, final_8b, mask

class GeometryManager:
    @staticmethod
    def standardize_laterality(img_8b, mask, lat, mac, mic):
        if lat.upper() == "L":
            w = img_8b.shape[1]
            def fl(b): return [(w - x2, y1, w - x1, y2) for x1, y1, x2, y2 in b]
            return np.fliplr(img_8b), np.fliplr(mask), fl(mac), fl(mic)
        return img_8b, mask, mac, mic

    @staticmethod
    def auto_roi_crop(img_8b, mask, margin=20):
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any() or not cols.any(): return img_8b, mask, 0, 0
        y0, y1 = max(0, int(np.where(rows)[0][0]) - margin), min(img_8b.shape[0], int(np.where(rows)[0][-1]) + margin)
        x0, x1 = max(0, int(np.where(cols)[0][0]) - margin), min(img_8b.shape[1], int(np.where(cols)[0][-1]) + margin)
        return img_8b[y0:y1, x0:x1], mask[y0:y1, x0:x1], x0, y0
