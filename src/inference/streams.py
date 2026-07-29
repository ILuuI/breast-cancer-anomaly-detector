"""
Estados y ejecucion de las dos vias del sistema Dual-Stream:

- MacroState / run_macro: analisis de contexto global (masas,
  asimetrias, distorsiones) sobre la imagen completa.
- MicroState / run_micro_smooth: barrido por ventana deslizante sobre
  parches de alta resolucion para deteccion de microcalcificaciones,
  consolidado mediante Supresion de No Maximos (NMS).
"""
import time
import cv2
import numpy as np

from config import device_arg, MICRO_BATCH as _MICRO_BATCH
from inference.nms import nms_fast
from preprocessing.dicom_processor import GeometryManager

class MacroState:
    __slots__ = ("detections", "done", "_reveal_t")
    def __init__(self): self.detections, self.done, self._reveal_t = [], False, None

class MicroState:
    __slots__ = ("raw_boxes", "final_boxes", "current_win", "wins_total", "wins_done", "done", "_reveal_t", "positions")
    def __init__(self):
        self.raw_boxes, self.final_boxes, self.current_win = [], [], None
        self.wins_total, self.wins_done, self.done, self._reveal_t, self.positions = 0, 0, False, None, []

def run_macro(img_8b, mask, model, conf, state):
    time.sleep(0.05) 
    c_8b, _, ox, oy = GeometryManager.auto_roi_crop(img_8b, mask)
    h_c, w_c = c_8b.shape; side = max(h_c, w_c)
    padded   = cv2.copyMakeBorder(c_8b, 0, side - h_c, 0, side - w_c, cv2.BORDER_CONSTANT, value=0)
    inp      = cv2.resize(padded, (640, 640), interpolation=cv2.INTER_AREA)
    results  = model.predict(cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR), conf=conf, device=device_arg, verbose=False)[0]
    scale    = 640 / side
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        state.detections.append((int(x1 / scale + ox), int(y1 / scale + oy), int(x2 / scale + ox), int(y2 / scale + oy), float(box.conf[0])))
    state.done = True

def run_micro_smooth(img_8b, mask, model, conf, state):
    H, W = img_8b.shape; psz, stride = 320, 160
    positions = []
    for y in range(0, H - psz + 1, stride):
        for x in range(0, W - psz + 1, stride):
            if np.mean(mask[y:y + psz, x:x + psz]) > 0.05: positions.append((x, y))
    state.positions, state.wins_total = positions, len(positions)
    if not positions: state.done = True; return

    for batch_start in range(0, len(positions), _MICRO_BATCH):
        batch_pos  = positions[batch_start : batch_start + _MICRO_BATCH]
        batch_imgs = []
        for (x, y) in batch_pos:
            patch = cv2.resize(img_8b[y:y + psz, x:x + psz], (640, 640), interpolation=cv2.INTER_CUBIC)
            batch_imgs.append(cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR))
        batch_results = model.predict(batch_imgs, conf=conf, device=device_arg, verbose=False)
        for (x, y), res in zip(batch_pos, batch_results):
            for box in res.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                state.raw_boxes.append((int(bx1 / 2 + x), int(by1 / 2 + y), int(bx2 / 2 + x), int(by2 / 2 + y), float(box.conf[0])))
        state.wins_done, state.current_win = batch_start + len(batch_pos), batch_pos[-1]
        time.sleep(0.001)

    state.current_win = None
    if state.raw_boxes:
        ab = np.array([[b[0], b[1], b[2], b[3]] for b in state.raw_boxes])
        ac = np.array([b[4] for b in state.raw_boxes])
        fb, fc = nms_fast(ab, ac, 0.2)
        state.final_boxes = [(b[0], b[1], b[2], b[3], c) for b, c in zip(fb, fc)]
    state.done = True

