"""Prepara el detector de vocabulario abierto de VigIA (YOLO-World) con los objetos de config.yaml.

Uso (una sola vez, con internet):   python scripts/preparar_vocabulario.py
Para reconocer otro objeto, agréguelo en config.yaml (modelos → vocabulario, nombre en inglés: nombre en
español) y vuelva a ejecutar este script. Después reinicie VigIA.

YOLO-World busca los objetos por su nombre escrito: un codificador de texto tipo CLIP convierte cada nombre en
un vector y el detector marca lo que se le parece en la imagen. El script guarda esos vectores dentro del
modelo, así VigIA no necesita internet ni CLIP mientras funciona.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from vigia.modelos import VOCABULARIO_BASE  # noqa: E402

PESOS = "yolov8s-worldv2.pt"      # versión "s": buen equilibrio entre precisión y velocidad en CPU


def main() -> int:
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))["modelos"]
    vocab = cfg.get("vocabulario") or VOCABULARIO_BASE
    destino = RAIZ / cfg.get("objetos_abierto", "models/vigia-vocabulario.pt")
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("Falta ultralytics: pip install -r requirements.txt")
        return 1
    os.chdir(destino.parent)                       # ultralytics descarga los pesos en la carpeta actual
    print(f"Cargando {PESOS} (la primera vez se descarga, ~25 MB)...")
    modelo = YOLOWorld(PESOS)
    nombres = list(vocab)
    print(f"Preparando {len(nombres)} objetos: " + ", ".join(f"{k} → {v}" for k, v in vocab.items()))
    try:
        modelo.set_classes(nombres)
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo preparar el vocabulario: {e}")
        print("Si dice que falta 'clip', instálelo y repita:  pip install git+https://github.com/ultralytics/CLIP.git")
        return 1
    modelo.save(str(destino))
    print(f"Listo: {destino}. Reinicie VigIA para usarlo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
