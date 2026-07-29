"""
Cliente del chatbot medico local (Ollama). Genera reportes clinicos en
texto plano a partir de los hallazgos de las vias Macro y Micro y
responde consultas del usuario en un hilo separado para no bloquear la
UI de OpenCV.
"""
import json
import queue
import threading
import urllib.request
import urllib.error

OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "llama3.2"

SYSTEM_PROMPT = (
    "Eres un asistente clinico especializado en interpretacion de mamografias. "
    "Tu funcion es analizar reportes de deteccion generados por modelos de IA (YOLO) "
    "para hallar masas y microcalcificaciones. Responde siempre en espanol, "
    "de forma concisa, tecnica y clinicamente responsable. "
    "No emitas diagnosticos definitivos; tu rol es asistir al radiologo. "
    "Cuando presentes el reporte inicial hazlo estructurado con secciones: "
    "HALLAZGOS MACRO, HALLAZGOS MICRO, IMPRESION DIAGNOSTICA, RECOMENDACION. "
    "Usa terminologia BI-RADS cuando sea apropiado."
)

class OllamaChatbot:
    """Chatbot medico que se comunica con Ollama en hilo separado."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model        = model
        self.history: list[dict] = []          # historial de mensajes
        self._out_q: queue.Queue = queue.Queue()  # cola de chunks hacia la UI
        self._busy        = False
        self._available   = False
        # Verificar disponibilidad de Ollama al inicio
        threading.Thread(target=self._check_ollama, daemon=True).start()

    # ── Verificacion de disponibilidad ──────────────────────────────────────
    def _check_ollama(self):
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                headers={"Content-Type": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def busy(self) -> bool:
        return self._busy

    # ── Enviar mensaje al modelo ─────────────────────────────────────────────
    def send(self, user_text: str):
        """Encola un mensaje de usuario y dispara la generacion en background."""
        if self._busy:
            return
        self.history.append({"role": "user", "content": user_text})
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        self._busy = True
        payload = json.dumps({
            "model":    self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            "stream":   True,
        }).encode()

        full_response = ""
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        full_response += token
                        # Quitar asteriscos (formato markdown)
                        token = token.replace("*", "").replace("#", "")
                        if token:
                            self._out_q.put(("token", token))
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError:
            self._out_q.put(("error", "[Ollama no disponible - verifique que este corriendo]"))
            full_response = "[Error de conexion con Ollama]"
        except Exception as e:
            msg = f"[Error: {str(e)[:60]}]"
            self._out_q.put(("error", msg))
            full_response = msg
        finally:
            # Guardar respuesta completa en historial
            clean = full_response.replace("*", "").replace("#", "")
            self.history.append({"role": "assistant", "content": clean})
            self._out_q.put(("done", clean))
            self._busy = False

    def poll(self):
        """Devuelve todos los chunks disponibles (no bloqueante)."""
        chunks = []
        while True:
            try:
                chunks.append(self._out_q.get_nowait())
            except queue.Empty:
                break
        return chunks

    def reset(self):
        """Limpia historial para un nuevo caso."""
        self.history.clear()
        while not self._out_q.empty():
            try: self._out_q.get_nowait()
            except queue.Empty: break
        self._busy = False


def build_clinical_report(img_id: str, lat: str, ms, mi, dcm) -> str:
    """Construye el reporte clinico estructurado para enviar a Ollama."""
    n_macro = len(ms.detections)
    n_micro = len(mi.final_boxes)

    # Descripcion de hallazgos macro
    macro_desc = []
    for i, (x1, y1, x2, y2, conf_v) in enumerate(ms.detections, 1):
        w_mm = x2 - x1; h_mm = y2 - y1
        macro_desc.append(
            f"  Hallazgo #{i}: region ({x1},{y1})-({x2},{y2}), "
            f"tamano aprox {w_mm}x{h_mm}px, confianza={conf_v:.2f}"
        )

    # Descripcion de hallazgos micro
    micro_desc = []
    for i, (x1, y1, x2, y2, conf_v) in enumerate(mi.final_boxes, 1):
        micro_desc.append(
            f"  Cluster #{i}: region ({x1},{y1})-({x2},{y2}), confianza={conf_v:.2f}"
        )

    # Metadata DICOM si esta disponible
    patient_age  = str(getattr(dcm, "PatientAge",  "N/D")).strip()
    patient_sex  = str(getattr(dcm, "PatientSex",  "N/D")).strip()
    study_date   = str(getattr(dcm, "StudyDate",   "N/D")).strip()
    modality     = str(getattr(dcm, "Modality",    "MG")).strip()
    institution  = str(getattr(dcm, "InstitutionName", "N/D")).strip()
    view_pos     = str(getattr(dcm, "ViewPosition", "N/D")).strip()

    lines = [
        f"=== REPORTE DE INFERENCIA IA - MAMOGRAFIA ===",
        f"Caso ID      : {img_id}",
        f"Lateralidad  : {lat} ({'Izquierda' if lat == 'L' else 'Derecha'})",
        f"Vista        : {view_pos}",
        f"Modalidad    : {modality}",
        f"Fecha Estudio: {study_date}",
        f"Institucion  : {institution}",
        f"Paciente - Edad: {patient_age} | Sexo: {patient_sex}",
        f"",
        f"--- MODELO GLOBAL (Deteccion de Masas / Anomalias Estructurales) ---",
        f"Total de hallazgos: {n_macro}",
    ]
    if macro_desc:
        lines += macro_desc
    else:
        lines.append("  Sin hallazgos macroscopicos detectados.")

    lines += [
        f"",
        f"--- MODELO LOCAL / SLIDING WINDOW (Microcalcificaciones) ---",
        f"Total de clusters: {n_micro}",
    ]
    if micro_desc:
        lines += micro_desc
    else:
        lines.append("  Sin microcalcificaciones detectadas.")

    lines += [
        f"",
        f"Analiza este reporte y presenta: hallazgos macro, hallazgos micro, "
        f"impresion diagnostica provisional (clasificacion BI-RADS sugerida) "
        f"y recomendacion clinica. Seras conciso y clinicamente responsable."
    ]

    return "\n".join(lines)
