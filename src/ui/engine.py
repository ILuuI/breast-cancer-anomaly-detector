"""
Motor grafico principal: gestiona la ventana de OpenCV, el frame
actual y los eventos de teclado/tick del loop de la aplicacion.
"""
import time
import threading
import cv2
import numpy as np

from ui.theme import C, WIN_NAME, pt, pt_center, draw_prog, _FONT

class Engine:
    def __init__(self):
        self._frame       = None
        self._lock        = threading.Lock()
        self.t0           = time.time()
        self.key          = -1
        self._wsize_cache = (1440, 860)
        self._wsize_ts    = 0.0
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1440, 860)

    @property
    def t(self): return time.time() - self.t0

    def wsize(self):
        now = time.time()
        if now - self._wsize_ts > 0.30:
            r = cv2.getWindowImageRect(WIN_NAME)
            if r[2] > 10 and r[3] > 10:
                self._wsize_cache = (max(800, r[2]), max(500, r[3]))
            self._wsize_ts = now
        return self._wsize_cache

    def blank(self, W=None, H=None):
        if W is None: W, H = self.wsize()
        f = np.empty((H, W, 3), dtype=np.uint8)
        f[:] = C.BG
        return f

    def push(self, frame):
        with self._lock: self._frame = frame

    def tick(self):
        with self._lock: f = self._frame
        if f is not None:
            W, H = self.wsize()
            if f.shape[0] != H or f.shape[1] != W: f = cv2.resize(f, (W, H))
            cv2.imshow(WIN_NAME, f)
        self.key = cv2.waitKeyEx(16)
        return self.key

    def header(self, f, phase, img_id="", prog=0.0):
        H, W = f.shape[:2]
        f[:48, :] = C.WHITE
        cv2.line(f, (0, 48), (W, 48), C.BORDER, 1)
        lx, ly = 22, 24
        cv2.circle(f, (lx, ly), 14, C.ACCENT, -1, cv2.LINE_AA)
        wave_pts = np.array([[lx-9,ly],[lx-5,ly],[lx-3,ly-7],[lx+1,ly+8],[lx+4,ly-5],[lx+7,ly],[lx+11,ly]], dtype=np.int32)
        cv2.polylines(f, [wave_pts], False, C.WHITE, 1, cv2.LINE_AA)
        pt(f, "MAMMO AI", (44, 29), 0.60, C.TEXT_HI, 2)
        cv2.circle(f, (44, 39), 3, C.GREEN, -1, cv2.LINE_AA)
        pt(f, "Asistente clinico", (51, 42), 0.30, C.GREEN, 1)
        pt_center(f, phase, 30, 0.46, C.TEXT_LO)
        if img_id:
            txt = f"ID: {img_id}"
            (tw, _), _ = cv2.getTextSize(txt, _FONT, 0.40, 1)
            pt(f, txt, (W - tw - 18, 30), 0.40, C.TEXT_LO, 1)
        draw_prog(f, 0, 46, W, 2, prog, C.ACCENT)

    def footer(self, f, msg=""):
        H, W = f.shape[:2]
        f[H - 28:H, :] = C.WHITE
        cv2.line(f, (0, H - 28), (W, H - 28), C.BORDER, 1)
        pt(f, msg, (15, H - 9), 0.38, C.TEXT_LO, 1)
