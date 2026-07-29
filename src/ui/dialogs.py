"""
Dialogos asincronos (seleccion de archivo/carpeta) usando tkinter en un
hilo separado, para no bloquear el loop principal de OpenCV.
"""
import queue
import threading

_dialog_queue: "queue.Queue[str]" = queue.Queue()
_dialog_active = False

def _dialog_worker(kind: str, initial_dir: str):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        if kind == "file":
            path = filedialog.askopenfilename(
                parent=root, title="Seleccionar DICOM",
                initialdir=initial_dir,
                filetypes=[("DICOM","*.dcm *.dicom *.DCM *.DICOM"),("Todos","*.*")])
        else:
            path = filedialog.askdirectory(parent=root, title="Seleccionar Carpeta", initialdir=initial_dir)
        root.destroy()
        _dialog_queue.put(path if path else "")
    except Exception:
        _dialog_queue.put("")

def open_dialog(kind="file", initial_dir="."):
    global _dialog_active
    _dialog_active = True
    threading.Thread(target=_dialog_worker, args=(kind, initial_dir), daemon=True).start()

def poll_dialog():
    global _dialog_active
    try:
        r = _dialog_queue.get_nowait()
        _dialog_active = False
        return r
    except queue.Empty:
        return None
