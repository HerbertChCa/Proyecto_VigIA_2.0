"""Encuesta de aceptación para la feria (página /encuesta).

Replica en Arequipa la pregunta de la PE3 del artículo: la aceptación de la videovigilancia inteligente es
condicional al propósito (82.2 % de aceptación con un fin protector, E45), al tipo de función (36 % para el
reconocimiento facial, E46) y a la confianza en quien gestiona los datos (E48). El corpus tenía un solo
estudio latinoamericano (E12): estas respuestas son datos propios para ese vacío.

Es anónima: no pide nombre, correo ni ningún dato que identifique a la persona, y no guarda su IP.
Solo responden mayores de edad.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

LIKERT = ["Muy en desacuerdo", "En desacuerdo", "Ni de acuerdo ni en desacuerdo", "De acuerdo", "Muy de acuerdo"]

# Cada pregunta: id, texto, tipo (likert | una | varias) y opciones. "ref" es la cifra del artículo con la que
# se compara el porcentaje de acuerdo (respuestas 4 y 5); los textos no son idénticos, es una referencia.
PREGUNTAS = [
    {"id": "general", "tipo": "likert", "bloque": "Aceptación",
     "texto": "Estoy de acuerdo con cámaras que detectan automáticamente situaciones de riesgo en espacios públicos "
              "para proteger a las personas.",
     "ref": {"valor": 82.2, "fuente": "E45: aceptación con un fin protector"}},
    {"id": "caidas", "tipo": "likert", "bloque": "Funciones", "texto": "Que detecten caídas y personas en el suelo."},
    {"id": "peleas", "tipo": "likert", "bloque": "Funciones", "texto": "Que detecten peleas o agresiones."},
    {"id": "robos", "tipo": "likert", "bloque": "Funciones", "texto": "Que detecten robos y asaltos."},
    {"id": "facial", "tipo": "likert", "bloque": "Funciones", "texto": "Que reconozcan quién soy (reconocimiento facial).",
     "ref": {"valor": 36.0, "fuente": "E46: aceptación del reconocimiento facial"}},
    {"id": "emociones", "tipo": "likert", "bloque": "Funciones", "texto": "Que analicen mis expresiones o emociones."},
    {"id": "esqueleto", "tipo": "una", "bloque": "Privacidad",
     "texto": "Si el sistema solo mostrara mi esqueleto y nunca mi imagen, mi aceptación…",
     "opciones": ["Aumenta", "Se mantiene igual", "Disminuye"]},
    {"id": "confianza", "tipo": "varias", "bloque": "Confianza",
     "texto": "¿En quién confiarías para gestionar estas cámaras? (puedes marcar varias)",
     "opciones": ["Municipalidad", "Policía Nacional", "Universidad", "Empresa privada de seguridad", "En ninguno"]},
    {"id": "lugares", "tipo": "varias", "bloque": "Lugares",
     "texto": "¿Dónde aceptarías este sistema? (puedes marcar varias)",
     "opciones": ["Calles y plazas", "Transporte público", "Universidad", "Centro de trabajo", "Tiendas y mercados",
                  "En ningún lugar"]},
    {"id": "rol", "tipo": "una", "bloque": "Sobre ti",
     "texto": "Eres…", "opciones": ["Estudiante de la USMP", "Estudiante de otra institución", "Docente o personal",
                                    "Público general"]},
    {"id": "edad", "tipo": "una", "bloque": "Sobre ti",
     "texto": "Tu edad", "opciones": ["18 a 24", "25 a 34", "35 a 49", "50 o más"]},
]
POR_ID = {p["id"]: p for p in PREGUNTAS}


def validar(respuestas: dict) -> dict:
    """Devuelve solo respuestas válidas; lanza ValueError si falta la confirmación de mayoría de edad."""
    if respuestas.get("mayor_de_edad") is not True:
        raise ValueError("La encuesta es solo para mayores de 18 años.")
    limpias = {}
    for p in PREGUNTAS:
        v = respuestas.get(p["id"])
        if v is None:
            continue
        if p["tipo"] == "likert" and isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5:
            limpias[p["id"]] = v
        elif p["tipo"] == "una" and v in p["opciones"]:
            limpias[p["id"]] = v
        elif p["tipo"] == "varias" and isinstance(v, list):
            elegidas = [o for o in p["opciones"] if o in v]
            if elegidas:
                limpias[p["id"]] = elegidas
    if not limpias:
        raise ValueError("No hay respuestas.")
    return limpias


class Encuesta:
    def __init__(self, carpeta: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(carpeta / "encuesta.db", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS respuestas (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "fecha TEXT, datos TEXT)")
        self._db.commit()

    def guardar(self, respuestas: dict) -> int:
        limpias = validar(respuestas)
        with self._lock:
            cur = self._db.execute("INSERT INTO respuestas (fecha, datos) VALUES (?, ?)",
                                   (datetime.now().isoformat(timespec="seconds"), json.dumps(limpias, ensure_ascii=False)))
            self._db.commit()
            return cur.lastrowid

    def _todas(self) -> list[tuple[str, dict]]:
        with self._lock:
            filas = self._db.execute("SELECT fecha, datos FROM respuestas ORDER BY id").fetchall()
        return [(f, json.loads(d)) for f, d in filas]

    def resultados(self) -> dict:
        todas = [d for _, d in self._todas()]
        out = {"n": len(todas), "preguntas": []}
        for p in PREGUNTAS:
            vals = [d[p["id"]] for d in todas if p["id"] in d]
            r = {"id": p["id"], "texto": p["texto"], "tipo": p["tipo"], "bloque": p["bloque"], "n": len(vals)}
            if p["tipo"] == "likert":
                r["conteo"] = [sum(1 for v in vals if v == k) for k in range(1, 6)]
                r["acuerdo"] = round(100 * sum(1 for v in vals if v >= 4) / len(vals), 1) if vals else None
                r["media"] = round(sum(vals) / len(vals), 2) if vals else None
                if "ref" in p:
                    r["ref"] = p["ref"]
            else:
                planas = [o for v in vals for o in (v if isinstance(v, list) else [v])]
                r["conteo"] = {o: planas.count(o) for o in p["opciones"]}
                # en «varias», el porcentaje es sobre las personas que respondieron esa pregunta
                r["porcentaje"] = {o: round(100 * c / len(vals), 1) if vals else 0.0 for o, c in r["conteo"].items()}
            out["preguntas"].append(r)
        return out

    def exportar_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["fecha"] + [p["id"] for p in PREGUNTAS])
        for fecha, d in self._todas():
            fila = [fecha]
            for p in PREGUNTAS:
                v = d.get(p["id"], "")
                fila.append("; ".join(v) if isinstance(v, list) else v)
            w.writerow(fila)
        return buf.getvalue()

    def cerrar(self) -> None:
        with self._lock:
            self._db.close()
