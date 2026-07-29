"""
Pantallas de carga inicial (phase0_loading), previsualizacion
asincrona de DICOM (AsyncPreviewWorker) y el explorador de archivos
DICOM (select_dicom_screen).
"""
import os
import glob
import math
import time
import threading
import cv2
import numpy as np
import pydicom

from ui.theme import C, pt, pt_center, draw_panel, draw_prog, draw_btn, fit_image, to_bgr, lerp
from ui.dialogs import open_dialog, poll_dialog

def phase0_loading(engine, load_fn):
    status = {"msg": "Inicializando Entorno Clinico...", "progress": 0.0, "done": False, "result": None, "error": None}
    def _worker():
        try:   status["result"] = load_fn(status)
        except Exception as e: status["error"] = str(e)
        finally: status["done"] = True
    threading.Thread(target=_worker, daemon=True).start()

    while not status["done"]:
        W, H = engine.wsize()
        f    = engine.blank(W, H)
        cy   = H // 2
        pt_center(f, "MAMMO AI", cy - 42, 1.2, C.TEXT_HI, 2)
        pt_center(f, "Sistema de Deteccion Dual-Stream v9.4", cy - 4, 0.44, C.TEXT_LO)
        pbw = min(400, W - 100); pbx = (W - pbw) // 2
        draw_prog(f, pbx, cy + 44, pbw, 4, status["progress"], C.ACCENT)
        pt_center(f, status["msg"], cy + 72, 0.40, C.TEXT_LO)
        engine.header(f, "PREPARANDO SISTEMA", "", status["progress"])
        engine.footer(f, "Cargando dependencias AI y parametros de validacion cruzada...")
        engine.push(f)
        if engine.tick() in (ord('q'), ord('Q')): return None
        time.sleep(0.016)
    return status["result"]

class AsyncPreviewWorker:
    def __init__(self):
        self.req_path, self.current_img, self.loading = None, None, False
        self.lock, self.alive = threading.Lock(), True
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def _run(self):
        last_processed = None
        while self.alive:
            with self.lock: path = self.req_path
            if path and path != last_processed:
                with self.lock: self.loading = True
                try:
                    dcm  = pydicom.dcmread(path)
                    raw  = dcm.pixel_array[::4, ::4].astype(np.float32)
                    photo = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
                    if photo == "MONOCHROME1": raw = raw.max() - raw + raw.min()
                    mn, mx = raw.min(), raw.max()
                    img  = np.clip((raw - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
                    img  = fit_image(to_bgr(img), 350, 500)
                    with self.lock: self.current_img = img
                except Exception:
                    with self.lock: self.current_img = "ERR"
                with self.lock: self.loading = False
                last_processed = path
            time.sleep(0.05)

    def request(self, path):
        with self.lock:
            if self.req_path != path: self.req_path, self.loading = path, True

    def get(self):
        with self.lock: return self.current_img, self.loading


def select_dicom_screen(engine, search_dir="."):
    candidates: list = []
    dialog_open, loading_dicom = False, False
    current_dir  = os.path.abspath(search_dir)
    previewer    = AsyncPreviewWorker()

    def _reload(d):
        nonlocal candidates, current_dir
        current_dir = os.path.abspath(d); c = []
        for ext in ["*.dicom", "*.dcm", "*.DCM", "*.DICOM"]:
            c += glob.glob(os.path.join(current_dir, "**", ext), recursive=True)
        candidates[:] = sorted(set(c))

    _reload(search_dir)
    selected, visible, scroll = 0, 14, 0

    while True:
        W, H   = engine.wsize()
        f      = engine.blank(W, H)
        ti     = engine.t
        engine.header(f, "SISTEMA DE ARCHIVOS", "", 0.0)

        cy2, ch2, lw = 60, H - 60 - 28, int(W * 0.60)
        draw_panel(f, 10, cy2, lw, ch2)
        pt(f, "ARCHIVOS DICOM DISPONIBLES", (25, cy2 + 35), 0.50, C.TEXT_HI, 2)
        pt(f, f"{len(candidates)} casos en: {current_dir[-60:]}" if candidates else f"Sin DICOMs en: {current_dir[-60:]}", (25, cy2 + 55), 0.40, C.TEXT_LO, 1)
        cv2.line(f, (20, cy2 + 65), (10 + lw - 10, cy2 + 65), C.BORDER, 1)

        rx, rw = 20 + lw, W - (20 + lw) - 10
        draw_panel(f, rx, cy2, rw, ch2, C.WHITE)
        pt(f, "PREVISUALIZACION CLINICA", (rx + 15, cy2 + 35), 0.50, C.TEXT_HI, 2)
        cv2.line(f, (rx + 10, cy2 + 65), (rx + rw - 10, cy2 + 65), C.BORDER, 1)

        if not candidates:
            pt_center(f, "Directorio vacio.", H // 2, 0.5, C.TEXT_LO)
        else:
            list_y, available_h = cy2 + 75, ch2 - 85
            row_h = max(24, min(34, available_h // visible))
            previewer.request(candidates[selected])

            for i in range(visible):
                idx = scroll + i
                if idx >= len(candidates): break
                is_sel = (idx == selected)
                ry2    = list_y + i * row_h
                if ry2 + row_h > cy2 + ch2 - 10: break
                if is_sel: draw_panel(f, 16, ry2 - 2, lw - 22, row_h, C.PANEL2, C.ACCENT)
                pt(f, os.path.basename(candidates[idx]), (26, ry2 + row_h // 2 + 4), 0.45, C.ACCENT if is_sel else C.TEXT_HI)

            p_img, is_loading = previewer.get()
            if is_loading:
                pulse = 0.5 + 0.5 * math.sin(ti * 6)
                pt(f, "Decodificando Matriz...", (rx + 40, cy2 + ch2 // 2), 0.45, lerp(C.TEXT_LO, C.ACCENT, pulse))
            elif isinstance(p_img, str) and p_img == "ERR":
                pt(f, "Formato DICOM Corrupto", (rx + 40, cy2 + ch2 // 2), 0.45, C.RED)
            elif p_img is not None:
                p_img = fit_image(p_img, rw - 20, ch2 - 95)
                ph, pw = p_img.shape[:2]
                pbx, pby = rx + (rw - pw) // 2, cy2 + 75 + (ch2 - 85 - ph) // 2
                f[pby:pby + ph, pbx:pbx + pw] = p_img
                cv2.rectangle(f, (pbx - 1, pby - 1), (pbx + pw, pby + ph), C.BORDER, 1)

        btn_bw, btn_bh = 280, 38
        btn_bx, btn_by = 10 + (lw - btn_bw) // 2, cy2 + ch2 - 50
        if dialog_open: draw_btn(f, btn_bx, btn_by, btn_bw, btn_bh, "Abriendo sistema...", False)
        else:           draw_btn(f, btn_bx, btn_by, btn_bw, btn_bh, "[ F ] Explorar Disco", True, C.TEXT_LO)

        if dialog_open:
            result = poll_dialog()
            if result is not None:
                dialog_open = False
                if result: _reload(result); selected, scroll = 0, 0

        if loading_dicom:
            H2, W2 = f.shape[:2]; ov = f.copy(); cv2.rectangle(ov, (0, 0), (W2, H2), C.BG, -1)
            cv2.addWeighted(ov, 0.85, f, 0.15, 0, f); cx_s, cy_s = W2 // 2, H2 // 2
            for di in range(8):
                angle = di * math.pi / 4 + ti * 4
                sx, sy = int(cx_s + 24 * math.cos(angle)), int(cy_s - 20 + 24 * math.sin(angle))
                a_dot = 0.2 + 0.8 * ((di / 8.0 + ti * 0.9) % 1.0)
                cv2.circle(f, (sx, sy), max(2, int(5 * a_dot)), lerp(C.TEXT_LO, C.ACCENT, a_dot), -1, cv2.LINE_AA)
            pt_center(f, "CARGANDO ESTUDIO DICOM", cy_s + 25, 0.55, C.TEXT_HI, 2)
            pt_center(f, "Iniciando pipeline analitico de alta resolucion...", cy_s + 45, 0.38, C.TEXT_LO)
            engine.footer(f, ""); engine.push(f); engine.tick(); time.sleep(0.016)
            previewer.alive = False; return candidates[selected]

        engine.footer(f, "Flechas: Navegar | ENTER/SPACE: Cargar Caso | F: Explorar Disco | Q: Salir")
        engine.push(f); k = engine.tick()
        if k in (ord('q'), ord('Q')): previewer.alive = False; return None
        elif k in (13, ord(' ')) and candidates: loading_dicom = True
        elif k in (2490368, 65362, 82):
            selected = max(0, selected - 1)
            if selected < scroll: scroll = selected
        elif k in (2621440, 65364, 84):
            selected = min(len(candidates) - 1, selected + 1)
            if selected >= scroll + visible: scroll = selected - visible + 1
        elif k in (ord('f'), ord('F')) and not dialog_open:
            open_dialog(kind="directory", initial_dir=current_dir); dialog_open = True
        time.sleep(0.016)

