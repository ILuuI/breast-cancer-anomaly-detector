"""
Dashboard clinico unificado: pantalla principal de la aplicacion.
Orquesta el preprocesamiento DICOM, dispara los hilos de inferencia
Macro/Micro, dibuja los resultados sobre la imagen, y aloja el panel
de chat con el asistente clinico (Ollama).
"""
import time
import threading
import math
import cv2
import numpy as np

try:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

from ui.theme import (
    C, pt, pt_center, pt_local, draw_panel, draw_prog, draw_btn,
    draw_glow_rect, draw_brackets, scan_line, fit_image, to_bgr, lerp, _badge,
    WIN_NAME, _FONT,
)
from ui.dialogs import open_dialog, poll_dialog
from preprocessing.dicom_processor import GeometryManager
from inference.streams import MacroState, MicroState, run_macro, run_micro_smooth
from chatbot.ollama_client import OllamaChatbot, build_clinical_report
from config import ANIM_SPEEDS, MICRO_ANIM_SPEED as _MICRO_ANIM_SPEED

def unified_analytical_dashboard(engine, dcm, processor, model_macro, model_micro, conf, img_id, lat, gt_macro, gt_micro):
    _fit_cache = {}
    try: after_mod = apply_modality_lut(dcm.pixel_array, dcm).astype(np.float32)
    except Exception: after_mod = dcm.pixel_array.astype(np.float32)
    try: raw = apply_voi_lut(after_mod, dcm).astype(np.float32)
    except Exception: raw = after_mod
    
    photo = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
    if photo == "MONOCHROME1": raw = raw.max() - raw + raw.min()

    steps, img_8b, mask = processor.process_steps(dcm)
    if img_8b is None: return "error"

    img_8b, mask, gt_macro, gt_micro = GeometryManager.standardize_laterality(img_8b, mask, lat, gt_macro, gt_micro)
    canvas_final = cv2.cvtColor(img_8b, cv2.COLOR_GRAY2BGR)

    def get_fitted(src_gray, tw, th):
        key = (id(src_gray), tw, th)
        if key not in _fit_cache: _fit_cache[key] = fit_image(to_bgr(src_gray), tw, th)
        return _fit_cache[key]

    sys_state, p_step, p_last_t = 0, 0, engine.t
    ms, mi = MacroState(), MicroState()
    inf_started, dialog_open = False, False

    zoom_state = {'s': 1.0, 'px': 0.0, 'py': 0.0, 'drag': False, 'mx': 0, 'my': 0, 'mx_drag': 0, 'my_drag': 0, 'active': False, 'cx': 0, 'cy': 0}
    _ui_anim_idx, _last_tick_t  = 0.0, engine.t

    # Configuración del Chatbot Integrado con Ollama
    bot = OllamaChatbot()
    chat_history = [
        ("AI", "Sistema listo. Cuando termine la inferencia, el reporte sera enviado automaticamente al asistente IA.")
    ]
    chat_input          = ""
    chatbot_triggered_report = False
    ai_buffer           = ""   # buffer de tokens en construccion
    ai_typing_done      = True  # True = IA no esta generando ahora mismo

    def mouse_cb(event, x, y, flags, param):
        zs = param; zs['mx'], zs['my'] = x, y
        if not zs['active']: return
        if event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_MOUSEHWHEEL:
            old_s = zs['s']
            zs['s'] = max(1.0, min(zs['s'] * (1.05 if flags > 0 else 1.0 / 1.05), 20.0))
            if zs['s'] <= 1.01: zs['px'], zs['py'] = 0.0, 0.0
            elif zs['cx'] > 0:
                dx, dy = x - zs['cx'], y - zs['cy']; sr = zs['s'] / old_s
                zs['px'] = dx - (dx - zs['px']) * sr; zs['py'] = dy - (dy - zs['py']) * sr
        elif event == cv2.EVENT_LBUTTONDOWN:
            zs['drag'] = True; zs['mx_drag'], zs['my_drag'] = x, y
        elif event == cv2.EVENT_LBUTTONUP: zs['drag'] = False
        elif event == cv2.EVENT_MOUSEMOVE and zs['drag']:
            zs['px'] += x - zs['mx_drag']; zs['py'] += y - zs['my_drag']
            zs['mx_drag'], zs['my_drag'] = x, y

    cv2.setMouseCallback(WIN_NAME, mouse_cb, zoom_state)

    while True:
        W, H   = engine.wsize()
        f      = engine.blank(W, H)
        ti     = engine.t

        cy2, ch2 = 48, H - 48 - 28
        info_w   = max(300, min(380, int(W * 0.30))) # Un poco más ancho para el Chatbot
        img_area_w = W - info_w - 14

        panel_ix, panel_iy = 4, cy2 + 2
        tw,  th            = img_area_w - 20, ch2 - 20
        ip_x, ip_y         = img_area_w + 10, cy2 + 2
        info_panel_w, info_panel_h = info_w - 4, ch2 - 4

        bg_col_panel = C.ZOOM_BG if sys_state == 2 else C.WHITE
        draw_panel(f, panel_ix, panel_iy, img_area_w - 4, ch2 - 4, bg_col_panel, C.BORDER)
        draw_panel(f, ip_x, ip_y, info_panel_w, info_panel_h, C.PANEL)

        # ──────────────────────────────────────────────────────────────────────
        # ESTADO 0 · TRANSICIÓN PROGRESIVA
        # ──────────────────────────────────────────────────────────────────────
        if sys_state == 0:
            n_steps  = len(steps); progress = p_step / n_steps
            engine.header(f, "PIPELINE PROCEDIMENTAL - PREPROCESAMIENTO", img_id, progress)
            step_dt = ti - p_last_t
            if step_dt > ANIM_SPEEDS["preproc_step_sec"]:
                p_step += 1; p_last_t = ti
                if p_step >= n_steps: sys_state = 1; continue

            cache_curr = (p_step, tw, th)
            if cache_curr not in _fit_cache: _fit_cache[cache_curr] = fit_image(to_bgr(steps[p_step][1]), tw, th)
            img_curr = _fit_cache[cache_curr]; rh, rw_ = img_curr.shape[:2]
            cfx, cfy = panel_ix + 10 + (tw - rw_) // 2, panel_iy + 10 + (th - rh) // 2

            alpha = min(1.0, step_dt / (ANIM_SPEEDS["preproc_step_sec"] * 0.6))
            if p_step > 0:
                cache_prev = (p_step - 1, tw, th); img_prev = _fit_cache.get(cache_prev, img_curr)
                f[cfy:cfy + rh, cfx:cfx + rw_] = cv2.addWeighted(img_curr, alpha, img_prev, 1.0 - alpha, 0) if img_prev.shape == img_curr.shape else img_curr
            else: f[cfy:cfy + rh, cfx:cfx + rw_] = img_curr

            scan_line(f, ti, C.ACCENT, 0.08, cfx, cfy, rw_, rh)
            pt(f, "PASOS COMPLETADOS DE ANATOMIA", (ip_x + 15, ip_y + 30), 0.45, C.TEXT_HI, 1)
            cv2.line(f, (ip_x + 10, ip_y + 45), (ip_x + info_panel_w - 10, ip_y + 45), C.BORDER)
            for i, (name, _) in enumerate(steps):
                ys = ip_y + 75 + i * 28
                col = (C.ACCENT if i == p_step else C.TEXT_HI if i < p_step else C.TEXT_LO)
                pt(f, f"[{'>>' if i == p_step else ('OK' if i < p_step else '--')}]", (ip_x + 15, ys), 0.40, col, 1)
                pt(f, name, (ip_x + 55, ys), 0.40, col, 1)

        # ──────────────────────────────────────────────────────────────────────
        # ESTADO 1 · INFERENCIA CONCURRENTE DUAL
        # ──────────────────────────────────────────────────────────────────────
        elif sys_state == 1:
            if not inf_started:
                threading.Thread(target=run_macro, args=(img_8b, mask, model_macro, conf, ms), daemon=True).start()
                threading.Thread(target=run_micro_smooth, args=(img_8b, mask, model_micro, conf, mi), daemon=True).start()
                inf_started = True

            half_w = (img_area_w - 4) // 2
            lx, ly = panel_ix, panel_iy
            draw_panel(f, lx, ly, half_w, ch2 - 4, C.ZOOM_BG, C.BORDER)
            pt(f, "MODELO GLOBAL", (lx + 12, ly + 22), 0.52, C.MACRO_COLOR, 1)
            pt(f, "YOLOv8 CBAM Full Image 640px", (lx + 12, ly + 40), 0.36, C.TEXT_LO, 1)
            if ms.done and not ms.detections:
                pt(f, "[ SIN HALLAZGOS ]", (lx + half_w - 150, ly + 30), 0.45, C.GREEN, 1)
            cv2.line(f, (lx + 10, ly + 60), (lx + half_w - 10, ly + 60), C.BORDER)

            fm = get_fitted(img_8b, half_w - 20, ch2 - 80); fmh, fmw = fm.shape[:2]; sv_m = fmw / img_8b.shape[1]
            mx_, my_ = lx + 10 + (half_w - 20 - fmw) // 2, ly + 70 + (ch2 - 80 - fmh) // 2
            f[my_:my_ + fmh, mx_:mx_ + fmw] = fm

            if not ms.done: scan_line(f, ti, C.MACRO_COLOR, 0.08, mx_, my_, fmw, fmh)
            else:
                if ms._reveal_t is None: ms._reveal_t = ti
                rp = min(1.0, (ti - ms._reveal_t) * 2.0)
                for di, (gx1, gy1, gx2, gy2, conf_v) in enumerate(ms.detections):
                    if min(1.0, max(0, rp * len(ms.detections) - di)) <= 0: continue
                    vx1, vy1 = max(mx_, min(int(gx1 * sv_m + mx_), mx_ + fmw - 1)), max(my_, min(int(gy1 * sv_m + my_), my_ + fmh - 1))
                    vx2, vy2 = max(mx_, min(int(gx2 * sv_m + mx_), mx_ + fmw - 1)), max(my_, min(int(gy2 * sv_m + my_), my_ + fmh - 1))
                    draw_glow_rect(f, vx1, vy1, vx2, vy2, C.MACRO_COLOR, 2)
                    _badge(f, max(mx_, vx1), max(my_, vy1), 46, 14, f"{conf_v:.2f}", tuple(int(c * 0.3) for c in C.MACRO_COLOR), C.WHITE)

            rx_mic = lx + half_w
            draw_panel(f, rx_mic, ly, half_w, ch2 - 4, C.ZOOM_BG, C.BORDER)
            pt(f, "MODELO LOCAL (SW)", (rx_mic + 12, ly + 22), 0.52, C.MICRO_COLOR, 1)
            pt(f, "Sliding Window 320px  |  Stride 160px", (rx_mic + 12, ly + 40), 0.36, C.TEXT_LO, 1)
            if mi.done and not mi.final_boxes:
                pt(f, "[ SIN HALLAZGOS ]", (rx_mic + half_w - 150, ly + 30), 0.45, C.GREEN, 1)
            cv2.line(f, (rx_mic + 10, ly + 60), (rx_mic + half_w - 10, ly + 60), C.BORDER)

            fmic = get_fitted(img_8b, half_w - 20, ch2 - 80); fmic_h, fmic_w = fmic.shape[:2]; sv_mi = fmic_w / img_8b.shape[1]
            mix_, miy_ = rx_mic + 10 + (half_w - 20 - fmic_w) // 2, ly + 70 + (ch2 - 80 - fmic_h) // 2
            f[miy_:miy_ + fmic_h, mix_:mix_ + fmic_w] = fmic

            for (x1, y1, x2, y2, _) in mi.raw_boxes[-50:]:
                cv2.rectangle(f, (int(x1 * sv_mi + mix_), int(y1 * sv_mi + miy_)), (int(x2 * sv_mi + mix_), int(y2 * sv_mi + miy_)), (40, 60, 80), 1)

            _dt = ti - _last_tick_t; _last_tick_t = ti
            if mi.positions and not mi.done:
                _ui_anim_idx = min(_ui_anim_idx + _dt * _MICRO_ANIM_SPEED, float(max(0, mi.wins_done - 1)))
                anim_idx = int(_ui_anim_idx)
                if 0 <= anim_idx < len(mi.positions):
                    wx, wy = mi.positions[anim_idx]
                    vwx1, vwy1 = int(wx * sv_mi + mix_), int(wy * sv_mi + miy_)
                    vwx2, vwy2 = int((wx + 320) * sv_mi + mix_), int((wy + 320) * sv_mi + miy_)
                    roi2 = f[vwy1:vwy2, vwx1:vwx2]
                    if roi2.size > 0:
                        ov3 = np.empty_like(roi2); ov3[:] = C.MICRO_COLOR
                        cv2.addWeighted(ov3, 0.07, roi2, 0.93, 0, roi2); f[vwy1:vwy2, vwx1:vwx2] = roi2
                    cv2.rectangle(f, (vwx1, vwy1), (vwx2, vwy2), lerp(C.BORDER, C.MICRO_COLOR, 0.5 + 0.5 * math.sin(ti * 8)), 1, cv2.LINE_AA)

            pt(f, f"Ventana: {mi.wins_done} / {mi.wins_total}", (rx_mic + 14, ly + ch2 - 75), 0.38, C.TEXT_LO, 1)

            if mi.done:
                if mi._reveal_t is None: mi._reveal_t = ti
                rp2 = min(1.0, (ti - mi._reveal_t) * 2.5)
                for di, (x1, y1, x2, y2, cv2_) in enumerate(mi.final_boxes):
                    if min(1.0, max(0, rp2 * len(mi.final_boxes) - di)) <= 0: continue
                    vx1m, vy1m = max(mix_, min(int(x1 * sv_mi + mix_), mix_ + fmic_w - 1)), max(miy_, min(int(y1 * sv_mi + miy_), miy_ + fmic_h - 1))
                    vx2m, vy2m = max(mix_, min(int(x2 * sv_mi + mix_), mix_ + fmic_w - 1)), max(miy_, min(int(y2 * sv_mi + miy_), miy_ + fmic_h - 1))
                    draw_glow_rect(f, vx1m, vy1m, vx2m, vy2m, C.MICRO_COLOR, 2)
                    _badge(f, max(mix_, vx1m), max(miy_, vy1m), 46, 14, f"{cv2_:.2f}", tuple(int(c * 0.28) for c in C.MICRO_COLOR), C.WHITE)

            pt(f, "ESTADO METODOLOGICO", (ip_x + 15, ip_y + 30), 0.45, C.TEXT_HI, 1)
            cv2.line(f, (ip_x + 10, ip_y + 45), (ip_x + info_panel_w - 10, ip_y + 45), C.BORDER)
            p_mac = 1.0 if ms.done else 0.5 + 0.5 * math.sin(ti * 3)
            p_mic = 1.0 if mi.done else (mi.wins_done / max(1, mi.wins_total))
            engine.header(f, "INFERENCIA DUAL-STREAM EN TIEMPO REAL", img_id, 0.50 + (p_mac * 0.25) + (p_mic * 0.25))

            pt_local(f, "Inferencia Macro (Global):", ip_x, ip_y + 80, info_panel_w, 0.42, C.TEXT_LO)
            draw_prog(f, ip_x + 15, ip_y + 92, info_panel_w - 30, 3, p_mac, C.GREEN if ms.done else C.MACRO_COLOR)
            pt_local(f, "COMPLETADO" if ms.done else "Procesando matriz...", ip_x, ip_y + 112, info_panel_w, 0.42, C.GREEN if ms.done else C.MACRO_COLOR, 1)

            pt_local(f, "Inferencia Micro (Local):", ip_x, ip_y + 150, info_panel_w, 0.42, C.TEXT_LO)
            draw_prog(f, ip_x + 15, ip_y + 162, info_panel_w - 30, 3, p_mic, C.GREEN if mi.done else C.MICRO_COLOR)
            pt_local(f, "COMPLETADO" if mi.done else "Analizando parches...", ip_x, ip_y + 182, info_panel_w, 0.42, C.GREEN if mi.done else C.MICRO_COLOR, 1)

            if ms.done and mi.done:
                sys_state = 2; zoom_state['active'] = True
                if gt_macro or gt_micro:
                    for x1, y1, x2, y2 in gt_macro: draw_brackets(canvas_final, x1, y1, x2, y2, C.GT_COLOR, 16, 4)
                    for x1, y1, x2, y2 in gt_micro: draw_brackets(canvas_final, x1, y1, x2, y2, C.GT_COLOR, 12, 3)
                for gx1, gy1, gx2, gy2, _ in ms.detections: draw_glow_rect(canvas_final, gx1, gy1, gx2, gy2, C.MACRO_COLOR, 5)
                for gx1, gy1, gx2, gy2, _ in mi.final_boxes: draw_glow_rect(canvas_final, gx1, gy1, gx2, gy2, C.MICRO_COLOR, 4)

        # ──────────────────────────────────────────────────────────────────────
        # ESTADO 2 · DIAGNÓSTICO INTERACTIVO — CHATBOT CLÍNICO CON OLLAMA
        # ──────────────────────────────────────────────────────────────────────
        elif sys_state == 2:
            engine.header(f, "DIAGNOSTICO ASISTIDO INTERACTIVO - OLLAMA AI", img_id, 1.0)
            zoom_state['cx'], zoom_state['cy'] = panel_ix + 10 + tw / 2, panel_iy + 10 + th / 2
            h_orig, w_orig = canvas_final.shape[:2]; s = min(tw / w_orig, th / h_orig) * zoom_state['s']

            max_px = max(0.0, (w_orig * s - tw) / 2 + 300); max_py = max(0.0, (h_orig * s - th) / 2 + 300)
            zoom_state['px'] = max(-max_px, min(max_px, zoom_state['px']))
            zoom_state['py'] = max(-max_py, min(max_py, zoom_state['py']))

            M = np.float32([[s, 0, (tw - w_orig * s) / 2 + zoom_state['px']], [0, s, (th - h_orig * s) / 2 + zoom_state['py']]])
            rendered_view = cv2.warpAffine(canvas_final, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=C.ZOOM_BG)
            cfx, cfy = panel_ix + 10, panel_iy + 10; f[cfy:cfy + th, cfx:cfx + tw] = rendered_view
            cv2.rectangle(f, (cfx - 1, cfy - 1), (cfx + tw, cfy + th), C.BORDER, 1)
            pt(f, f"Ampliacion: {zoom_state['s']:.2f}x", (cfx + 15, cfy + 28), 0.44, C.ACCENT, 1)

            # ── ENCABEZADO CHATBOT ─────────────────────────────────────────────
            status_dot = C.GREEN if bot.available else C.RED
            cv2.circle(f, (ip_x + info_panel_w - 20, ip_y + 22), 5, status_dot, -1, cv2.LINE_AA)
            status_lbl = "Ollama OK" if bot.available else "Ollama OFF"
            pt(f, status_lbl, (ip_x + info_panel_w - 80, ip_y + 27), 0.32,
               C.GREEN if bot.available else C.RED, 1)
            pt(f, "ASISTENTE CLINICO IA", (ip_x + 15, ip_y + 25), 0.50, C.ACCENT, 2)
            cv2.line(f, (ip_x + 10, ip_y + 35), (ip_x + info_panel_w - 10, ip_y + 35), C.BORDER)

            # ── AUTO-ENVIO DEL REPORTE AL TERMINAR INFERENCIA ─────────────────
            if not chatbot_triggered_report:
                chatbot_triggered_report = True
                ai_typing_done = False
                ai_buffer      = ""
                if bot.available:
                    report_prompt = build_clinical_report(img_id, lat, ms, mi, dcm)
                    chat_history.append(("User", f"[Reporte automatico generado para caso {img_id}]"))
                    bot.send(report_prompt)
                else:
                    fallback = (
                        f"[Ollama no disponible] Reporte local: Caso {img_id} | "
                        f"Macro: {len(ms.detections)} hallazgo(s) | "
                        f"Micro: {len(mi.final_boxes)} cluster(s). "
                        "Inicie Ollama para analisis IA completo."
                    )
                    chat_history.append(("AI", fallback))
                    ai_typing_done = True

            # ── POLLING DE TOKENS DESDE OLLAMA ────────────────────────────────
            chunks = bot.poll()
            for ctype, cval in chunks:
                if ctype == "token":
                    ai_buffer += cval
                    ai_typing_done = False
                elif ctype in ("done", "error"):
                    # Mensaje completo listo: agregarlo al historial
                    full_msg = ai_buffer.strip() if ai_buffer.strip() else cval
                    chat_history.append(("AI", full_msg))
                    ai_buffer      = ""
                    ai_typing_done = True

            # ── AREA DEL HISTORIAL DE CHAT ─────────────────────────────────────
            chat_box_y = ip_y + 45
            chat_box_h = info_panel_h - 160
            draw_panel(f, ip_x + 10, chat_box_y, info_panel_w - 20, chat_box_h, C.PANEL2, C.BORDER)

            # Construir lista de mensajes a mostrar
            display_msgs = list(chat_history)
            # Si la IA esta generando, mostrar buffer parcial como mensaje en curso
            if not ai_typing_done and ai_buffer:
                partial = ai_buffer + ("_" if int(engine.t * 3) % 2 == 0 else " ")
                display_msgs.append(("AI", partial))

            curr_y = chat_box_y + 14
            max_lines_px = chat_box_h - 10
            # Calcular lineas totales para scroll automatico (mostrar las ultimas)
            rendered_lines = []
            for sender, msg in display_msgs:
                prefix = "Dr: " if sender == "User" else "AI: "
                col_txt = C.TEXT_HI if sender == "User" else C.GREEN
                words = msg.split()
                line = prefix
                for wrd in words:
                    wrd = wrd.replace("*", "").replace("#", "")
                    test = line + " " + wrd
                    (lw_t, _), _ = cv2.getTextSize(test, _FONT, 0.36, 1)
                    if lw_t > info_panel_w - 45:
                        rendered_lines.append((line, col_txt))
                        line = "    " + wrd
                    else:
                        line = test
                rendered_lines.append((line, col_txt))
                rendered_lines.append(("", col_txt))  # separador

            line_h = 16
            max_visible = max_lines_px // line_h
            visible_lines = rendered_lines[-max_visible:] if len(rendered_lines) > max_visible else rendered_lines
            for txt_line, col_txt in visible_lines:
                if curr_y > chat_box_y + chat_box_h - 4: break
                if txt_line:
                    pt(f, txt_line, (ip_x + 20, curr_y), 0.36, col_txt, 1)
                curr_y += line_h

            # ── INDICADOR "GENERANDO..." ───────────────────────────────────────
            if not ai_typing_done:
                dots = "." * (int(engine.t * 3) % 4)
                pt(f, f"IA generando{dots}", (ip_x + 20, chat_box_y + chat_box_h - 14),
                   0.33, C.ACCENT, 1)

            # ── INPUT TEXTBOX ──────────────────────────────────────────────────
            input_y = chat_box_y + chat_box_h + 10
            input_border = C.TEXT_LO if bot.busy else C.ACCENT
            draw_panel(f, ip_x + 10, input_y, info_panel_w - 20, 36, C.WHITE, input_border)
            hint = "Esperando respuesta IA..." if bot.busy else "Escriba consulta + ENTER..."
            pt(f, hint, (ip_x + 15, input_y - 6), 0.32, C.TEXT_LO, 1)
            display_input = chat_input + ("|" if int(engine.t * 2) % 2 == 0 else "")
            pt(f, display_input, (ip_x + 18, input_y + 22), 0.38, C.TEXT_HI, 1)

            # ── BOTONES DE ACCION ──────────────────────────────────────────────
            btn_w = (info_panel_w - 30) // 2
            b1x, b2x = ip_x + 10, ip_x + 20 + btn_w
            by_actions = info_panel_h + ip_y - 48
            draw_btn(f, b1x, by_actions, btn_w, 36, "NUEVO CASO", True, C.ACCENT)
            draw_btn(f, b2x, by_actions, btn_w, 36, "CERRAR SESION", True, C.TEXT_LO)

        # ──────────────────────────────────────────────────────────────────────
        # EVENTOS GLOBALES Y ATAJOS DE TECLADO / ENTRADA CHAT
        # ──────────────────────────────────────────────────────────────────────
        engine.footer(f, "Chat: Escriba + ENTER | Touchpad: Zoom | Click+Arrastre: Paneo | O: Nuevo Caso | Q: Salir")
        engine.push(f)
        k = engine.tick()

        if sys_state == 2:
            if k == 13: # ENTER enviado en Chat
                if chat_input.strip() and not bot.busy:
                    user_q = chat_input.strip()
                    chat_history.append(("User", user_q))
                    chat_input    = ""
                    ai_typing_done = False
                    ai_buffer      = ""
                    if bot.available:
                        bot.send(user_q)
                    else:
                        # Fallback local si Ollama no esta disponible
                        q_lower = user_q.lower()
                        if "macro" in q_lower or "global" in q_lower:
                            ans = f"El Analisis Macro reporta {len(ms.detections)} sospechas estructurales densas."
                        elif "micro" in q_lower or "calcificacion" in q_lower:
                            ans = f"El Analisis Micro detecto {len(mi.final_boxes)} microcalcificaciones (NMS aplicado)."
                        elif "reporte" in q_lower or "resumen" in q_lower:
                            ans = f"Caso {img_id}: Macro={len(ms.detections)}, Micro={len(mi.final_boxes)}."
                        else:
                            ans = "[Ollama no disponible] Inicie Ollama para respuestas IA completas."
                        chat_history.append(("AI", ans))
                        ai_typing_done = True
            elif k == 8: # Backspace
                chat_input = chat_input[:-1]
            elif k == 32:  # Espacio — consumido AQUI, nunca llega al dialogo
                if len(chat_input) < 80:
                    chat_input += ' '
            elif 33 <= k <= 126 and len(chat_input) < 80:  # resto de printables (sin espacio)
                chat_input += chr(k)
            elif k in (ord('+'), ord('=')): zoom_state['s'] = min(20.0, zoom_state['s'] * 1.08)
            elif k in (ord('-'), ord('_')): 
                zoom_state['s'] = max(1.0, zoom_state['s'] / 1.08)
                if zoom_state['s'] <= 1.01: zoom_state['px'], zoom_state['py'] = 0.0, 0.0
            elif k in (ord('w'), ord('W'), 2490368): zoom_state['py'] += 35 
            elif k in (ord('s'), ord('S'), 2621440): zoom_state['py'] -= 35 
            elif k in (ord('a'), ord('A'), 2424832): zoom_state['px'] += 35 
            elif k in (ord('d'), ord('D'), 2555904): zoom_state['px'] -= 35

        if k in (ord('q'), ord('Q')):
            cv2.setMouseCallback(WIN_NAME, lambda *a: None); return "quit", None

        if sys_state == 2 and k in (ord('o'), ord('O')):
            bot.reset()
            open_dialog(kind="file", initial_dir="."); dialog_open = True

        if sys_state == 2 and dialog_open:
            result = poll_dialog()
            if result is not None:
                dialog_open = False
                if result:
                    cv2.setMouseCallback(WIN_NAME, lambda *a: None); _fit_cache.clear()
                    return "file", result

        time.sleep(0.016)

