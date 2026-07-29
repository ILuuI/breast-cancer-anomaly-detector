"""
Paleta clinica, primitivas de renderizado vectorial y utilidades OpenCV
compartidas por todas las pantallas de la aplicacion.

IMPORTANTE: todo el texto que se dibuja en pantalla debe pasar por
ascii_safe()/pt() para evitar caracteres especiales no soportados por
las fuentes de OpenCV (ver .gitignore / notas de proyecto).
"""
import math
import cv2
import numpy as np

class C:
    BG          = (245, 247, 250)
    PANEL       = (255, 255, 255)
    PANEL2      = (230, 235, 242)
    BORDER      = (200, 210, 222)
    ACCENT      = (210, 120,  40)  # Azul clínico corporativo (BGR)
    GREEN       = (180, 130,  30)  # Azul claro / Medio informativo
    GT_COLOR    = (150, 100,  20)  # Azul marino profundo
    MACRO_COLOR = (230, 140,  20)  # Azul brillante macro
    MICRO_COLOR = (190,  90,  10)  # Azul oscuro micro
    TEXT_HI     = ( 40,  48,  56)
    TEXT_LO     = (130, 140, 152)
    RED         = ( 60,  70, 180)  # Mantenido solo para alertas críticas de error
    WHITE       = (255, 255, 255)
    ZOOM_BG     = ( 15,  20,  25)  

WIN_NAME = "MAMMO_AI_CLINICAL_SYSTEM"
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVAS DE RENDERIZADO VECTORIAL (Tipografía limpia)
# ══════════════════════════════════════════════════════════════════════════════
def ascii_safe(text: str) -> str:
    """Convierte caracteres especiales al equivalente ASCII para OpenCV."""
    REPLACEMENTS = {
        'á':'a','é':'e','í':'i','ó':'o','ú':'u',
        'Á':'A','É':'E','Í':'I','Ó':'O','Ú':'U',
        'ñ':'n','Ñ':'N','ü':'u','Ü':'U',
        '\u2013':'-','\u2014':'-','\u2018':"'",'\u2019':"'",
        '\u201c':'"','\u201d':'"','…':'...','°':'deg',
    }
    return ''.join(REPLACEMENTS.get(c, c) for c in text)

def pt(img, text, pos, scale=0.5, color=C.TEXT_HI, thick=1):
    cv2.putText(img, ascii_safe(text), pos, _FONT, scale, color, thick, cv2.LINE_AA)

def pt_center(img, text, cy, scale=0.55, color=C.TEXT_HI, thick=1):
    W = img.shape[1]
    text = ascii_safe(text)
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thick)
    pt(img, text, (max(0, (W - tw) // 2), cy), scale, color, thick)

def pt_local(img, text, x, y, max_w, scale=0.42, color=C.TEXT_HI, thick=1):
    cv2.putText(img, ascii_safe(text), (x, y), _FONT, scale, color, thick, cv2.LINE_AA)

def draw_panel(img, x, y, w, h, color=C.PANEL, border=C.BORDER, alpha=1.0):
    if w <= 0 or h <= 0: return
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x2 <= x or y2 <= y: return
    if alpha < 1.0:
        roi = img[y:y2, x:x2]
        ov  = np.empty_like(roi); ov[:] = color
        cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)
    else:
        cv2.rectangle(img, (x, y), (x2, y2), color, -1)
    if border is not None:
        cv2.rectangle(img, (x, y), (x2 - 1, y2 - 1), border, 1)

def draw_prog(img, x, y, w, h, p, fg=C.ACCENT, bg=C.PANEL2):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * max(0.0, min(1.0, p)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)

def lerp(c1, c2, t): return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_glow_rect(img, x1, y1, x2, y2, color, thick=2):
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), C.WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

def draw_brackets(img, x1, y1, x2, y2, color, sz=16, thick=2):
    for (px, py), (dx, dy) in zip(
            [(x1, y1), (x2, y1), (x1, y2), (x2, y2)],
            [(1, 1),   (-1, 1),  (1, -1),  (-1, -1)]):
        cv2.line(img, (px, py), (px + dx * sz, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * sz), color, thick, cv2.LINE_AA)

def fit_image(img, tw, th):
    h, w = img.shape[:2]
    if h == 0 or w == 0: return img
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def to_bgr(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

def norm8(arr):
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-6: return np.zeros_like(a, dtype=np.uint8)
    return np.clip((a - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)

def draw_btn(img, x, y, w, h, text, active=True, color=None):
    col = (color if color is not None else C.ACCENT) if active else C.BORDER
    bg  = C.BG if active else C.PANEL2
    cv2.rectangle(img, (x, y), (x + w, y + h), bg,  -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col,  1)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
    pt(img, text, (x + (w - tw) // 2, y + (h + th) // 2), 0.45, col if active else C.TEXT_LO, 1)

def scan_line(img, t, color=C.ACCENT, alpha=0.12, bx=0, by=0, bw=None, bh=None):
    h, w  = img.shape[:2]
    ex    = min((bx + bw) if bw is not None else w, w)
    ey    = min((by + bh) if bh is not None else h, h)
    roi = img[by:ey, bx:ex]
    if roi.size == 0: return
    col_arr = np.array(color, dtype=np.float32)
    y     = by + int((math.sin(t * 2.2) * 0.5 + 0.5) * max(1, ey - by))
    y     = max(by, min(ey - 1, y))
    beam_h = 20
    y1_local = max(0, (y - by) - beam_h)
    y2_local = min(ey - by, y - by + 1)
    if y2_local > y1_local:
        sub_roi = roi[y1_local:y2_local, :]
        alphas = np.linspace(0, alpha, y2_local - y1_local).reshape(-1, 1, 1)
        blended = np.clip(sub_roi.astype(np.float32) * (1.0 - alphas) + col_arr * alphas, 0, 255).astype(np.uint8)
        roi[y1_local:y2_local, :] = blended
    cv2.line(img, (bx, y), (ex, y), color, 1, cv2.LINE_AA)

def _badge(img, x, y, w, h, text, bg_col, tx_col):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_col, -1)
    pt(img, text, (x + 4, y + h - 3), 0.35, tx_col, 1)

