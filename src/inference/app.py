"""
Orquestador principal de la aplicacion MamoIA: carga los modelos YOLO
(macro/micro), gestiona el login, la seleccion de DICOM y el ciclo del
dashboard clinico.
"""
import os
import cv2
import pandas as pd
import pydicom
import torch
from ultralytics import YOLO

from preprocessing.dicom_processor import PixelMonotonicProcessor
from ui.engine import Engine
from ui.login import run_login_screen
from ui.screens import phase0_loading, select_dicom_screen
from ui.dashboard import unified_analytical_dashboard

class ThesisInferencerVisual:
    def __init__(self, macro_weights, micro_weights, csv_path):
        self.macro_w, self.micro_w, self.csv_path = macro_weights, micro_weights, csv_path
        self.processor = PixelMonotonicProcessor()
        self.engine    = Engine()
        self.model_macro = self.model_micro = self.df = None

    def _load_models(self, status):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        status["msg"] = f"Desplegando Arquitectura YOLO Macro ({self.device.upper()})..."; status["progress"] = 0.15
        self.model_macro = YOLO(self.macro_w)
        self.model_macro.to(self.device)

        status["msg"] = f"Desplegando Arquitectura YOLO Micro ({self.device.upper()})..."; status["progress"] = 0.55
        self.model_micro = YOLO(self.micro_w)
        self.model_micro.to(self.device)

        status["msg"] = "Indexando Datos de Validacion Cientifica..."; status["progress"] = 0.85
        if os.path.exists(self.csv_path): self.df = pd.read_csv(self.csv_path)
        status["msg"] = "Estructura lista. Abriendo Dashboard Clinico."; status["progress"] = 1.0
        return True

    def get_gt(self, img_id):
        if self.df is None: return [], []
        df2 = self.df[self.df["image_id"] == img_id]; mac, mic = [], []
        for _, row in df2.iterrows():
            cats = str(row["finding_categories"])
            if "No Finding" in cats or pd.isna(row["xmin"]): continue
            box = [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]
            (mic if "Calcification" in cats else mac).append(box)
        return mac, mic

    def run(self, dicom_path=None, conf=0.25, search_dir="."):
        # Ejecutar el Login antes de desplegar modelos y procesar
        if not run_login_screen(self.engine):
            cv2.destroyAllWindows(); return

        if phase0_loading(self.engine, self._load_models) is None:
            cv2.destroyAllWindows(); return

        current_dicom = dicom_path
        while True:
            if current_dicom is None:
                current_dicom = select_dicom_screen(self.engine, search_dir)
                if current_dicom is None: break

            img_id = os.path.splitext(os.path.basename(current_dicom))[0]
            try: dcm = pydicom.dcmread(current_dicom)
            except Exception: current_dicom = None; continue

            lat = str(getattr(dcm, "ImageLaterality", "R")).strip().upper()
            mac_gt, mic_gt = self.get_gt(img_id)

            action = unified_analytical_dashboard(
                self.engine, dcm, self.processor,
                self.model_macro, self.model_micro, conf,
                img_id, lat, mac_gt, mic_gt)

            if action == "error":     current_dicom = None
            elif action[0] == "quit": break
            elif action[0] == "file": current_dicom = action[1]

        cv2.destroyAllWindows()
