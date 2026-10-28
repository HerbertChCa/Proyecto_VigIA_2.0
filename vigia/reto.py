"""Retos para la feria (página /reto): el visitante actúa una situación y VigIA intenta detectarla.

Cada intento termina en «VigIA te detectó» o «no te detectó». Además de atraer al público, el conteo de
intentos detectados es una medida en vivo de la sensibilidad (recall) de cada regla con personas reales,
que es lo que el artículo pide reportar en lugar de cifras de laboratorio (brecha de validación, Tabla 11).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

RETOS = [
    {"id": "caida", "titulo": "Simula una caída",
     "instruccion": "Camina frente a la cámara y déjate caer con cuidado sobre la colchoneta. Quédate en el suelo.",
     "tipos": ["caida", "en_suelo"], "limite": 25},
    {"id": "intrusion", "titulo": "Entra a la zona prohibida",
     "instruccion": "Camina hasta pisar la zona roja marcada en el piso.",
     "tipos": ["intrusion"], "limite": 20, "zona": "restringida"},
    {"id": "merodeo", "titulo": "Merodea",
     "instruccion": "Quédate dando vueltas dentro de la zona amarilla, como si vigilaras el lugar.",
     "tipos": ["merodeo"], "limite": 30, "zona": "merodeo"},
    {"id": "abandono", "titulo": "Abandona una mochila",
     "instruccion": "Deja una mochila en el piso, a la vista de la cámara, y aléjate de ella.",
     "tipos": ["abandono"], "limite": 35},
    {"id": "hurto", "titulo": "Llévate la mochila de otro (en pareja)",
     "instruccion": "Uno deja su mochila en el piso y se aleja; el otro se acerca y se la lleva.",
     "tipos": ["hurto"], "limite": 50},
    {"id": "sonrisa", "titulo": "Sonríe a la cámara",
     "instruccion": "Mira de frente a la cámara y sonríe. Requiere el análisis de expresiones.",
     "emocion": "alegría", "limite": 15},
]
POR_ID = {r["id"]: r for r in RETOS}

# Tiempos cortos para que cada reto dure segundos y no minutos (se aplican con --feria).
TIEMPOS_FERIA = {"merodeo.tiempo_max": 10, "objetos.tiempo_abandono": 8, "caida.en_suelo_alerta": 3}


class Marcador:
    """Intentos y detecciones por reto, guardados en datos/reto.json (sin nombres ni imágenes)."""

    def __init__(self, archivo: Path):
        self.archivo = archivo
        self._lock = threading.Lock()
        self.datos = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}

    def registrar(self, reto: str, detectado: bool, segundos: float | None) -> dict:
        if reto not in POR_ID:
            raise ValueError(f"reto desconocido: {reto}")
        with self._lock:
            d = self.datos.setdefault(reto, {"intentos": 0, "detectados": 0, "tiempos": []})
            d["intentos"] += 1
            if detectado:
                d["detectados"] += 1
                if segundos is not None and 0 <= segundos <= 600:
                    d["tiempos"].append(round(float(segundos), 1))
            self.archivo.write_text(json.dumps(self.datos, ensure_ascii=False, indent=2), encoding="utf-8")
            return self.resumen()

    def resumen(self) -> dict:
        out = {}
        for r in RETOS:
            d = self.datos.get(r["id"], {"intentos": 0, "detectados": 0, "tiempos": []})
            t = sorted(d["tiempos"])
            out[r["id"]] = {
                "intentos": d["intentos"], "detectados": d["detectados"],
                "tasa": round(100 * d["detectados"] / d["intentos"], 1) if d["intentos"] else None,
                "mediana_s": t[len(t) // 2] if t else None,
            }
        return out
