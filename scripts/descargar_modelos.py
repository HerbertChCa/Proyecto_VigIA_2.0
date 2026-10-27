"""Descarga los modelos en la carpeta models/. Uso: python scripts/descargar_modelos.py"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "models"

ONNX = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    "facial_expression_recognition_mobilefacenet_2022july.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx",
        "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/facial_expression_recognition_mobilefacenet_2022july.onnx",
    ],
}
YOLO_PESOS = ["yolo11n-pose.pt", "yolo11n.pt"]


def valido(ruta: Path) -> bool:
    if not ruta.exists() or ruta.stat().st_size < 50_000:
        return False
    with open(ruta, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            return False      # se bajó el puntero de Git LFS, no el modelo
    try:
        import cv2
        cv2.dnn.readNet(str(ruta))
        return True
    except Exception:
        return False


def bajar_onnx():
    for nombre, urls in ONNX.items():
        destino = MODELOS / nombre
        if valido(destino):
            print(f"  ya existe {nombre}")
            continue
        for url in urls:
            try:
                print(f"  descargando {nombre} desde {url.split('/')[2]} ...")
                urllib.request.urlretrieve(url, destino)
                if valido(destino):
                    print(f"  listo ({destino.stat().st_size // 1024} KB)")
                    break
            except Exception as e:
                print(f"    falló: {e}")
        else:
            destino.unlink(missing_ok=True)
            print(f"  AVISO: no se pudo descargar {nombre}. El sistema funciona igual: difumina la cabeza "
                  "usando la pose y desactiva las emociones.")


def bajar_yolo():
    from ultralytics import YOLO
    for nombre in YOLO_PESOS:
        destino = MODELOS / nombre
        if destino.exists():
            print(f"  ya existe {nombre}")
            continue
        print(f"  descargando {nombre} ...")
        YOLO(nombre)                      # ultralytics lo baja en la carpeta actual
        if Path(nombre).exists():
            shutil.move(nombre, destino)
        print("  listo")


if __name__ == "__main__":
    MODELOS.mkdir(exist_ok=True)
    print("Modelos de rostro y expresión (OpenCV Zoo):")
    bajar_onnx()
    print("Modelos YOLO11 (Ultralytics):")
    try:
        bajar_yolo()
    except Exception as e:
        print(f"  ERROR: {e}\n  Revise la conexión a internet y que ultralytics esté instalado.")
        sys.exit(1)
