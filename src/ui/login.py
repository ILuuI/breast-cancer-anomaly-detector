"""
Pantalla de login (autenticacion basica de facultativo) previa al
despliegue de los modelos y al dashboard de inferencia.
"""
import time
import cv2

from ui.theme import C, pt, pt_center, draw_panel, draw_btn

def run_login_screen(engine):
    username, password = "", ""
    active_field = 0 # 0: Usuario, 1: Contraseña, 2: Botón
    error_msg = ""
    
    while True:
        W, H = engine.wsize()
        f = engine.blank(W, H)
        
        # Fondo decorativo médico suave
        cv2.circle(f, (W, 0), int(W*0.4), C.PANEL2, -1, cv2.LINE_AA)
        cv2.circle(f, (0, H), int(H*0.3), C.PANEL2, -1, cv2.LINE_AA)
        
        # Tarjeta de login central
        lw, lh = 420, 360
        lx, ly = (W - lw) // 2, (H - lh) // 2
        draw_panel(f, lx, ly, lw, lh, C.PANEL, C.BORDER)
        
        pt_center(f, "MAMMO AI CLINICAL PORTAL", ly + 40, 0.65, C.TEXT_HI, 2)
        pt_center(f, "Ingrese sus credenciales de facultativo", ly + 65, 0.38, C.TEXT_LO, 1)
        
        # Input: Usuario
        u_col = C.ACCENT if active_field == 0 else C.BORDER
        draw_panel(f, lx + 40, ly + 110, lw - 80, 42, C.WHITE, u_col)
        pt(f, "Usuario:", (lx + 45, ly + 102), 0.38, C.TEXT_LO, 1)
        pt(f, username + ("|" if active_field == 0 and int(engine.t*2)%2==0 else ""), (lx + 55, ly + 136), 0.48, C.TEXT_HI, 1)
        
        # Input: Contraseña
        p_col = C.ACCENT if active_field == 1 else C.BORDER
        draw_panel(f, lx + 40, ly + 185, lw - 80, 42, C.WHITE, p_col)
        pt(f, "Contrasena:", (lx + 45, ly + 177), 0.38, C.TEXT_LO, 1)
        stars = "*" * len(password)
        pt(f, stars + ("|" if active_field == 1 and int(engine.t*2)%2==0 else ""), (lx + 55, ly + 211), 0.48, C.TEXT_HI, 1)
        
        # Botón Ingresar
        btn_col = C.TEXT_HI if active_field == 2 else C.ACCENT
        draw_btn(f, lx + 40, ly + 260, lw - 80, 45, "INICIAR SESION", True, btn_col)
        
        if error_msg:
            pt_center(f, error_msg, ly + 330, 0.40, C.RED, 1)
            
        engine.header(f, "AUTENTICACION REQUERIDA", "", 0.0)
        engine.footer(f, "TAB / Flechas: Cambiar Campo | ENTER: Confirmar | Esc: Salir")
        engine.push(f)
        
        k = engine.tick()
        if k == 27: # ESC
            return False
        elif k in (9, 2621440, 65364, 84): # TAB / Flecha Abajo
            active_field = (active_field + 1) % 3
            error_msg = ""
        elif k in (2490368, 65362, 82): # Flecha Arriba
            active_field = (active_field - 1) % 3
            error_msg = ""
        elif k == 13: # ENTER
            if active_field == 2 or active_field == 1:
                # Simulación básica de validación facultativa
                if username.strip() != "" and len(password) >= 4:
                    return True
                else:
                    error_msg = "Credenciales invalidas o muy cortas."
                    active_field = 0
            else:
                active_field = 1
        elif k == 8: # Backspace
            if active_field == 0: username = username[:-1]
            elif active_field == 1: password = password[:-1]
        elif 32 <= k <= 126:
            if active_field == 0 and len(username) < 20: username += chr(k)
            elif active_field == 1 and len(password) < 20: password += chr(k)
            
        time.sleep(0.016)
