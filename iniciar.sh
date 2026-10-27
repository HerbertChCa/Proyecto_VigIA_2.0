#!/usr/bin/env bash
# VigIA - inicio en Linux o macOS. Uso: ./iniciar.sh   o   ./iniciar.sh --demo
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [ ! -f .venv/instalado.txt ]; then
  pip install --upgrade pip && pip install -r requirements.txt && echo ok > .venv/instalado.txt
fi
if [ "$1" != "--demo" ] && [ ! -f models/yolo11n-pose.pt ]; then python scripts/descargar_modelos.py; fi
python -m vigia "$@"
