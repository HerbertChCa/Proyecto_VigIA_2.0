"""Exporta los modelos YOLO de VigIA a OpenVINO para que el análisis en vivo corra más rápido en CPU.

En una laptop sin tarjeta gráfica, PyTorch es la forma más lenta de correr YOLO. OpenVINO es el motor de Intel
para CPU (funciona también en procesadores AMD) y suele ser 2 a 3 veces más rápido con el mismo modelo y
las mismas detecciones: no cambia la precisión, solo el tiempo que tarda cada cuadro.

Se corre una vez, con internet (la primera vez instala el paquete openvino):

  python scripts\\acelerar_modelos.py              exporta con los tamaños de config.yaml
  python scripts\\acelerar_modelos.py --rapido     exporta con los tamaños del perfil rápido (feria)

Crea, junto a cada .pt, una carpeta `<modelo>_openvino_model/`. VigIA la usa sola al arrancar (lo dice en la
consola: «personas en openvino»). Para volver a PyTorch: `aceleracion: "no"` en config.yaml, o borre las carpetas.
Un modelo exportado solo acepta el tamaño de imagen con el que se exportó: si cambia `tamano_imagen` o
`tamano_imagen_objetos` en config.yaml, vuelva a correr este script.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.modelos import modelo_abierto, ruta_acelerada, ruta_modelo  # noqa: E402
from vigia.rendimiento import aplicar_perfil  # noqa: E402


def exportar(ruta: str, tarea: str, imgsz: int) -> bool:
    from ultralytics import YOLO
    print(f"\n- {Path(ruta).name} ({tarea}, {imgsz} px)")
    vieja = ruta_acelerada(ruta)
    if vieja is not None:
        shutil.rmtree(vieja)                    # se rehace: pudo exportarse con otro tamaño
    try:
        t0 = time.time()
        destino = YOLO(ruta).export(format="openvino", imgsz=imgsz, half=False, dynamic=False)
        print(f"  listo en {time.time() - t0:.0f} s -> {destino}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  NO se pudo exportar ({e}). VigIA seguirá usando el .pt para este modelo.")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--rapido", action="store_true", help="usar los tamaños del perfil rápido")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    aplicar_perfil(cfg, "rapido" if a.rapido else cfg.get("procesamiento", {}).get("perfil", "normal"))
    m = cfg["modelos"]
    tam_obj = m.get("tamano_imagen_objetos", m["tamano_imagen"])
    ok = exportar(ruta_modelo(m["pose"]), "pose", m["tamano_imagen"])
    abierto = modelo_abierto(m)
    if abierto:
        ok &= exportar(abierto, "detect", tam_obj)
    ok &= exportar(ruta_modelo(m["objetos"]), "detect", tam_obj)   # el de COCO: respaldo si se apaga el vocabulario
    print("\nListo. Arranque VigIA como siempre; en la consola debe decir «personas en openvino»."
          if ok else "\nAlgún modelo no se exportó; VigIA usa el .pt para ese y la versión rápida para los demás.")
    print("Para comparar la velocidad: python scripts\\medir_velocidad.py <video.mp4>")


if __name__ == "__main__":
    main()
