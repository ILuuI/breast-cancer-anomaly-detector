# MamoIA — Sistema Multimodal de Inteligencia Artificial para el Apoyo Diagnóstico de Cáncer de Mama

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RT--DETR%20%2F%20YOLO-purple)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey?logo=windows)](#)

Sistema de apoyo al diagnóstico de cáncer de mama mediante visión artificial, desarrollado como proyecto de investigación en la **Universidad de Sucre** en colaboración con el **Hospital Universitario de Sincelejo**. La aplicación de escritorio detecta macro y micro calcificaciones en mamografías digitales (DICOM) utilizando un pipeline de inferencia dual y genera reportes clínicos automáticos apoyados en un modelo de lenguaje local.

## Descripción general

El cáncer de mama es el más frecuente en mujeres a nivel mundial, y en el departamento de Sucre el acceso a especialistas en radiología con experiencia sigue siendo limitado. MamoIA busca ofrecer una herramienta accesible que apoye el proceso diagnóstico, actuando como un segundo lector automatizado sin reemplazar el criterio médico ni emitir diagnósticos de malignidad, mediante:

- Arquitectura **Dual-Stream**: una vía Macro para masas, asimetrías y distorsiones arquitecturales, y una vía Micro especializada en microcalcificaciones mediante parches de alta resolución.
- **Pixel Monotonic Processor**: pipeline de estandarización radiológica de imágenes DICOM (Modality LUT, VOI LUT, segmentación anatómica, winsorización y mapeo de tono monotónico) que preserva la fidelidad física de las intensidades antes de cuantizar a 8 bits.
- Mecanismo de atención **CBAM** (Convolutional Block Attention Module) integrado antes de las cabezas de detección para mejorar la selectividad frente a tejido fibroglandular denso.
- Integración de ambas vías mediante barrido por ventana deslizante con solapamiento y **Supresión de No Máximos (NMS)** para consolidar detecciones en coordenadas globales sin duplicidades.
- Generación automática de **reportes clínicos en texto plano** mediante un chatbot local (Ollama).
- Registro de usuarios y trazabilidad de estudios mediante **SQLite**.

## Arquitectura del pipeline

```
DICOM (mamografía) → Pixel Monotonic Processor → Inferencia Dual-Stream
                                                     ├── Vía Macro: YOLOv8m + CBAM (imagen completa 640x640)
                                                     └── Vía Micro: YOLOv8m base (ventana deslizante, parches 320x320 → 640x640)
                                                              ↓
                                          Fusión (NMS) + Reporte clínico (LLM local)
```

Backbone de detección validado mediante estudio de ablación (YOLOv8m, YOLOv8m+CBAM, YOLOv11m, YOLOv11m+CBAM); en evolución hacia **RT-DETR-L** (Ultralytics) y modelos especializados en mamografía como **Mammo-CLIP** (EfficientNet-B5) y **Mammo-FM**.

## Dataset

- **VinDr-Mammo**: repositorio multicéntrico de 5,000 estudios de mamografía digital de campo completo (20,000 imágenes DICOM), con anotaciones de tipo caja delimitadora para hallazgos BI-RADS 3, 4 y 5.
- Muestra de trabajo: 5,028 imágenes procesadas — 3,260 sanas y 1,768 con hallazgos (1,482 con lesiones macroscópicas y 442 con microcalcificaciones).
- Vía Micro expandida mediante jittering (factor 3) a 1,326 muestras positivas, más ~9,780 parches de negativos duros filtrados por control de calidad (fracción de tejido, rango de intensidad y coeficiente de variación).
- Consolidación de anotaciones múltiples mediante Weighted Boxes Fusion (WBF, IoU=0.5).

## Metodología

- **Preprocesamiento (Pixel Monotonic Processor)**: transformación Modality LUT + VOI LUT según estándar DICOM (NEMA PS3.3), segmentación anatómica del tejido mamario (filtro Gaussiano 19x19 + umbral adaptativo + cierre morfológico), winsorización robusta por percentiles (0.1–99.9%), mapeo de tono monotónico (combinación potencia-sigmoide, γ=1.8) y cuantización final a 8 bits.
- **Vía Macro**: imagen completa redimensionada a 640x640 con zero-padding; excluye anotaciones de calcificaciones para reducir ruido.
- **Vía Micro**: parches nativos de 320x320 centrados en lesiones, reescalados a 640x640 (zoom digital); aumentación por jittering (factor 3) y minería de negativos duros con control de calidad de parches.
- **Entrenamiento**: transferencia de aprendizaje desde pesos MS COCO, optimizador AdamW (lr=1e-3), hasta 120 épocas con parada temprana (tolerancia 30 épocas), ejecutado en Google Colab.
- **Métricas**: Precisión, Recall y mAP@0.5 / mAP@0.5:0.95 sobre detección de cajas delimitadoras.

## Estructura del repositorio

```
MamoIA/
├── src/
│   ├── main.py                       # Punto de entrada de la aplicación
│   ├── config.py                      # Configuración global (device, velocidades UI)
│   ├── ui/
│   │   ├── theme.py                   # Paleta clínica y primitivas de dibujo OpenCV
│   │   ├── engine.py                  # Motor gráfico (ventana, loop, eventos)
│   │   ├── dialogs.py                  # Diálogos de archivo (tkinter async)
│   │   ├── login.py                    # Pantalla de autenticación
│   │   ├── screens.py                  # Carga inicial + selector de DICOM
│   │   └── dashboard.py                 # Dashboard clínico + panel de chat
│   ├── inference/
│   │   ├── nms.py                      # Supresión de No Máximos
│   │   ├── streams.py                   # Vía Macro / Vía Micro (dual-stream)
│   │   └── app.py                       # Orquestador (carga modelos, ciclo principal)
│   ├── preprocessing/
│   │   └── dicom_processor.py           # Pixel Monotonic Processor + GeometryManager
│   ├── chatbot/
│   │   └── ollama_client.py             # Cliente Ollama + generación de reportes
│   └── db/                               # Modelo de datos SQLite (pendiente)
├── models/                                # Pesos entrenados (ver sección de descarga)
├── data/
│   ├── raw/                               # Dataset VinDr-Mammo (no versionado)
│   └── annotations/                        # Anotaciones convertidas
├── notebooks/                               # Entrenamiento en Google Colab
├── docs/                                     # Propuesta, poster y documentación académica
├── pictures/                                  # Capturas de la app y resultados
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Instalación (Windows 10/11)

```bash
git clone https://github.com/imauroo2023-sudo/Sistema-Multimodal-de-cancer-de-mama.git
cd Sistema-Multimodal-de-cancer-de-mama
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Descarga de pesos entrenados

Los pesos del modelo (`PESOS_MACRO`, `PESOS_MICRO`) no se versionan en este repositorio por su tamaño. Están alojados en Hugging Face Hub:

> **Modelo en Hugging Face**: [huggingface.co/tadashidetemu/mamoia-dualstream-yolov8](https://huggingface.co/tadashidetemu/mamoia-dualstream-yolov8)

**Opción A — descarga manual:**

Entra al link de arriba, pestaña *Files and versions*, descarga `PESOS_MACRO.pt` y `PESOS_MICRO.pt`, y colócalos en `models/`.

**Opción B — descarga por código (recomendado):**

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

macro_path = hf_hub_download(repo_id="tadashidetemu/mamoia-dualstream-yolov8", filename="PESOS_MACRO.pt", local_dir="models")
micro_path = hf_hub_download(repo_id="tadashidetemu/mamoia-dualstream-yolov8", filename="PESOS_MICRO.pt", local_dir="models")
```

### Ejecución

Configura las rutas locales de pesos, CSV de anotaciones y carpeta de DICOM mediante variables de entorno (o edita las constantes por defecto en `src/main.py`):

```bash
set MAMOIA_PESOS_MACRO=C:\ruta\a\macro.pt
set MAMOIA_PESOS_MICRO=C:\ruta\a\micro.pt
set MAMOIA_CSV_ANOTACIONES=C:\ruta\a\finding_annotations.csv
set MAMOIA_SEARCH_DIR=C:\ruta\a\dicoms

python src/main.py
```

## Resultados

Estudio de ablación en dos fases (arquitectura y aumentación de datos), entrenado sobre VinDr-Mammo con transferencia de aprendizaje desde pesos MS COCO, optimizador AdamW (lr=1e-3), máximo 120 épocas con parada temprana (30 épocas de tolerancia).

### Fase I — Ablación arquitectónica (sin aumentación)

**Vía Macro**

| Arquitectura | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8m (Base) | 0.5189 | 0.3620 | 0.3619 | 0.1912 |
| **YOLOv8m + CBAM** | **0.5296** | **0.3942** | **0.4107** | **0.2199** |
| YOLOv11m (Base) | 0.4641 | 0.3675 | 0.3634 | 0.1917 |
| YOLOv11m + CBAM | 0.5331 | 0.3383 | 0.3906 | 0.2060 |

**Vía Micro**

| Arquitectura | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **YOLOv8m (Base)** | **0.6771** | **0.6344** | **0.6670** | **0.3968** |
| YOLOv8m + CBAM | 0.6423 | 0.5985 | 0.6110 | 0.3786 |
| YOLOv11m (Base) | 0.6137 | 0.6263 | 0.6416 | 0.3943 |
| YOLOv11m + CBAM | 0.6574 | 0.5968 | 0.6518 | 0.3716 |

**Conclusión de la Fase I**: CBAM mejora consistentemente la Vía Macro (mejor resultado con YOLOv8m+CBAM, mAP@0.5=0.4107) al ayudar a ignorar el estroma denso; en la Vía Micro, el YOLOv8m base sin atención adicional es superior para detectar patrones de alta frecuencia como las microcalcificaciones.

### Fase II — Impacto de la aumentación de datos offline

**Vía Macro** (sobre las configuraciones + CBAM)

| Arquitectura | Fase | Precisión | Recall | mAP@0.5 |
|---|---|---|---|---|
| YOLOv8m + CBAM | I (sin aumentación) | 0.5296 | 0.3942 | 0.4107 |
| YOLOv8m + CBAM | II (con aumentación) | 0.4919 | 0.3851 | 0.3680 |
| YOLOv11m + CBAM | I (sin aumentación) | 0.5331 | 0.3383 | 0.3906 |
| YOLOv11m + CBAM | II (con aumentación) | 0.4999 | 0.4149 | 0.3828 |

**Vía Micro** (sobre las configuraciones base)

| Arquitectura | Fase | Precisión | Recall | mAP@0.5 |
|---|---|---|---|---|
| YOLOv8m (Base) | I (sin aumentación) | 0.6771 | 0.6344 | 0.6670 |
| YOLOv8m (Base) | II (con aumentación) | 0.6157 | 0.5728 | 0.6487 |
| YOLOv11m (Base) | I (sin aumentación) | 0.6137 | 0.6263 | 0.6416 |
| YOLOv11m (Base) | II (con aumentación) | 0.6044 | 0.4375 | 0.5195 |

**Conclusión de la Fase II**: bajo las restricciones computacionales del entorno de entrenamiento (Google Colab, recursos limitados), la aumentación de datos no mejoró el desempeño general — probablemente por convergencia temprana debido al menor número de iteraciones disponibles para los modelos más pesados. Estos resultados se consideran preliminares.

### Configuración final integrada

- **Vía Macro**: YOLOv8m + CBAM (mejor equilibrio precisión/recall/localización).
- **Vía Micro**: YOLOv8m base (mejor desempeño sin necesidad de atención adicional).
- Integración vía ventana deslizante con solapamiento + NMS para consolidar detecciones sin duplicidades en coordenadas globales.

> Nota: estas métricas corresponden al estudio de ablación (VinDr-Mammo, YOLOv8m/v11m con y sin CBAM). Si entrenas con RT-DETR-L u otro backbone, actualiza esta tabla con tus propios resultados.

## Limitaciones y trabajo futuro

- Validación clínica pendiente con el Hospital Universitario de Sincelejo.
- Evaluación comparativa entre YOLO, RT-DETR y modelos especializados (Mammo-CLIP, Mammo-FM).
- Ampliación del dataset con casos locales del departamento de Sucre.

## Autores

- Mauro y Sara — Universidad de Sucre

## Colaboración institucional

Hospital Universitario de Sincelejo.

## Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE).
