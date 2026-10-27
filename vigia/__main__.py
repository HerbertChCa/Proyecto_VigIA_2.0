"""Punto de entrada: python -m vigia [--demo] [--config config.yaml]"""
from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path

import uvicorn
import yaml

from .servidor import RAIZ, crear_app


def main():
    ap = argparse.ArgumentParser(description="VigIA: análisis de comportamiento en cámaras de seguridad")
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="escena simulada, sin cámara ni modelos")
    ap.add_argument("--datos", default=str(RAIZ / "datos"), help="carpeta de alertas, capturas y ajustes")
    ap.add_argument("--sin-navegador", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    datos = Path(a.datos)
    if a.demo:
        datos = datos / "demo"
    app = crear_app(cfg, datos, demo=a.demo)
    host, puerto = cfg["servidor"]["host"], cfg["servidor"]["puerto"]
    if not a.sin_navegador:
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{host}:{puerto}")).start()
    uvicorn.run(app, host=host, port=puerto, log_level="warning")


if __name__ == "__main__":
    main()
