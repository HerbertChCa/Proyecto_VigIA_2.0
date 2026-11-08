"""Punto de entrada: python -m vigia [--demo] [--config config.yaml]"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import webbrowser
from pathlib import Path

import uvicorn
import yaml

# PyTorch y OpenCV usan por defecto todos los núcleos en cada inferencia; con varias cámaras eso satura la CPU
# y el panel se congela. Se deja libre un núcleo para el servidor web y el navegador. Debe fijarse antes de
# importar torch (lo importa ultralytics al cargar los modelos).
HILOS = max(1, (os.cpu_count() or 2) - 1)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(HILOS))

from .rendimiento import aplicar_perfil  # noqa: E402
from .servidor import RAIZ, crear_app, ip_local  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="VigIA: análisis de comportamiento en cámaras de seguridad")
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="escena simulada, sin cámara ni modelos")
    ap.add_argument("--datos", default=str(RAIZ / "datos"), help="carpeta de alertas, capturas y ajustes")
    ap.add_argument("--sin-navegador", action="store_true")
    ap.add_argument("--feria", action="store_true",
                    help="modo feria: retos con tiempos cortos y encuesta abierta a los celulares de la red local")
    ap.add_argument("--rapido", action="store_true",
                    help="perfil rápido: imágenes más chicas para analizar más cuadros por segundo en vivo (feria)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    perfil = aplicar_perfil(cfg, "rapido" if a.rapido else cfg.get("procesamiento", {}).get("perfil"))
    if perfil != "normal":
        print(f"Perfil de rendimiento: {perfil}")
    datos = Path(a.datos)
    if a.demo:
        datos = datos / "demo"
    app = crear_app(cfg, datos, demo=a.demo, feria=a.feria)
    host, puerto = cfg["servidor"]["host"], cfg["servidor"]["puerto"]
    if a.feria:
        host = "0.0.0.0"           # los celulares de la misma red abren /encuesta; el resto sigue siendo solo local
        print(f"Modo feria: encuesta en http://{ip_local()}:{puerto}/encuesta  ·  retos en http://127.0.0.1:{puerto}/reto")
    if not a.sin_navegador:
        local = "127.0.0.1" if host == "0.0.0.0" else host
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{local}:{puerto}" + ("/reto" if a.feria else ""))).start()
    # al cerrar con el navegador abierto, uvicorn cancela esas conexiones: no es un error que deba mostrarse
    logging.getLogger("uvicorn.error").addFilter(
        lambda r: not ((r.exc_info and r.exc_info[0] is asyncio.CancelledError) or "graceful shutdown" in r.getMessage()))
    print("VigIA en marcha. Para cerrarlo pulse Ctrl+C (si no responde, púlselo otra vez).")
    # el video en vivo y los eventos son conexiones que no terminan solas: sin un límite, Ctrl+C esperaría
    # a que el navegador las cierre y la terminal quedaría colgada
    try:
        uvicorn.run(app, host=host, port=puerto, log_level="warning", timeout_graceful_shutdown=2)
    except KeyboardInterrupt:       # uvicorn >= 0.29 vuelve a lanzar el Ctrl+C después de cerrar el servidor
        pass
    print("VigIA detenido.")
    os._exit(0)     # corta los hilos de las cámaras que sigan esperando a una transmisión caída


if __name__ == "__main__":
    main()
