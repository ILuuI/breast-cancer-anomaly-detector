"""
Punto de entrada de MamoIA — Dual-Stream Mammography Inference System.

Uso:
    python src/main.py [ruta_opcional_a_un_dicom]

Las rutas de pesos, CSV de anotaciones y carpeta de busqueda de DICOM
se configuran abajo. Ajusta estas constantes a tu entorno local antes
de ejecutar, o expon las mismas rutas como variables de entorno si
prefieres no tener rutas absolutas en el codigo versionado.
"""
import sys
import os

from inference.app import ThesisInferencerVisual

# ──────────────────────────────────────────────────────────────────────────
# RUTAS DE CONFIGURACION — ajustar a tu entorno local
# ──────────────────────────────────────────────────────────────────────────
SEARCH_DIR      = os.environ.get("MAMOIA_SEARCH_DIR", r"E:\cancer\chatbot\dicom")
PESOS_MACRO     = os.environ.get("MAMOIA_PESOS_MACRO", r"E:\cancer\chatbot\data\macro.pt")
PESOS_MICRO     = os.environ.get("MAMOIA_PESOS_MICRO", r"E:\cancer\chatbot\data\micro.pt")
CSV_ANOTACIONES = os.environ.get("MAMOIA_CSV_ANOTACIONES", r"E:\cancer\chatbot\data\finding_annotations.csv")


def main():
    dicom_arg = sys.argv[1] if len(sys.argv) > 1 else None
    motor = ThesisInferencerVisual(PESOS_MACRO, PESOS_MICRO, CSV_ANOTACIONES)
    motor.run(dicom_path=dicom_arg, conf=0.25, search_dir=SEARCH_DIR)


if __name__ == "__main__":
    main()
