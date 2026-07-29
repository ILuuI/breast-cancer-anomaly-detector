"""
Configuracion global compartida por la aplicacion: velocidades de
animacion de la UI y deteccion del dispositivo de computo (GPU/CPU)
para los modelos YOLO.
"""
import torch

ANIM_SPEEDS = {
    "preproc_step_sec": 0.50,
    "macro_scan_period": 1.2,
    "micro_cells_sec":  75.0,
}

MICRO_BATCH = 8
MICRO_ANIM_SPEED = 20.0

# 0 = primera GPU disponible, "cpu" si no hay CUDA
device_arg = 0 if torch.cuda.is_available() else "cpu"
