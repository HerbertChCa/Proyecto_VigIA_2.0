"""Muestra qué objetos ve el detector en una imagen o en el primer segundo de un video.

Uso:   python scripts/probar_objetos.py ruta\\a\\foto.jpg      (o un .mp4)
Sirve para comprobar el vocabulario antes de la feria y ajustar las confianzas en config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from vigia.modelos import CLASES_OBJETO, modelo_abierto, ruta_modelo, vocabulario  # noqa: E402


def main(ruta: str) -> int:
    from ultralytics import YOLO
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))["modelos"]
    abierto = modelo_abierto(cfg)
    if abierto:
        modelo, vocab = YOLO(abierto), vocabulario(cfg)
        mapa, conf, clases = {int(i): vocab.get(n, n) for i, n in modelo.names.items()}, 0.15, None
        print("Detector: vocabulario abierto")
    else:
        modelo, mapa, conf, clases = YOLO(ruta_modelo(cfg["objetos"])), CLASES_OBJETO, 0.30, list(CLASES_OBJETO)
        print("Detector: COCO (no se encontró el vocabulario abierto; ejecute scripts/preparar_vocabulario.py)")
    cap = cv2.VideoCapture(ruta)
    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer", ruta)
        return 1
    r = modelo.predict(frame, imgsz=cfg["tamano_imagen"], conf=conf, classes=clases, verbose=False)[0]
    filas = sorted(((float(c), mapa.get(int(k), "?"), b) for b, k, c in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())), reverse=True)
    for c, nombre, b in filas:
        print(f"  {nombre:<12} {c:5.2f}   caja {[int(v) for v in b]}")
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 255), 2)
        cv2.putText(frame, f"{nombre} {c:.2f}", (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    salida = Path(ruta).with_name(Path(ruta).stem + "_objetos.jpg")
    cv2.imwrite(str(salida), frame)
    print(f"{len(filas)} objetos. Imagen marcada: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) > 1 else print(__doc__) or 1)
