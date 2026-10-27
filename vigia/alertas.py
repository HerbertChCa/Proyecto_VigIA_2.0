"""Gestor de alertas: evita duplicados, guarda evidencia, registra el veredicto del operador
y avisa al panel en tiempo real.

El veredicto (confirmada o falsa alarma) responde a dos vacíos del artículo: ningún estudio evalúa
sus explicaciones con operadores y solo 2 de 44 miden el desempeño fuera del conjunto de
entrenamiento (Tabla 11). Con él, cada instalación obtiene su propia tasa de falsas alarmas.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .tipos import Evento

NOMBRES = {
    "violencia": "Violencia", "caida": "Caída", "en_suelo": "Persona en el suelo", "merodeo": "Merodeo",
    "intrusion": "Intrusión", "abandono": "Objeto abandonado", "hurto": "Posible hurto", "emocion": "Emoción",
}
VEREDICTOS = ("confirmada", "falsa")
COLUMNAS = ["id", "fecha", "camara", "tipo", "severidad", "mensaje", "explicacion", "captura", "atendida", "veredicto"]


class GestorAlertas:
    def __init__(self, carpeta: Path, enfriamiento: float = 30.0):
        self.carpeta = carpeta
        (carpeta / "capturas").mkdir(parents=True, exist_ok=True)
        self.enfriamiento = enfriamiento
        self._ultima: dict[str, float] = {}
        self._lock = threading.Lock()
        self._db = sqlite3.connect(carpeta / "alertas.db", check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, camara TEXT, tipo TEXT, severidad TEXT,
            mensaje TEXT, explicacion TEXT, captura TEXT, atendida INTEGER DEFAULT 0, veredicto TEXT)""")
        cols = {f[1] for f in self._db.execute("PRAGMA table_info(alertas)")}
        if "veredicto" not in cols:      # bases creadas con la versión anterior
            self._db.execute("ALTER TABLE alertas ADD COLUMN veredicto TEXT")
        self._db.commit()
        self._suscriptores: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    # ---- suscripción del panel (Server-Sent Events)
    def suscribir(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._suscriptores.append((asyncio.get_running_loop(), q))
        return q

    def desuscribir(self, q: asyncio.Queue) -> None:
        self._suscriptores = [(l, x) for l, x in self._suscriptores if x is not q]

    def _difundir(self, datos: dict) -> None:
        for loop, q in list(self._suscriptores):
            try:
                loop.call_soon_threadsafe(q.put_nowait, datos)
            except Exception:
                pass

    # ---- registro
    def procesar(self, camara: str, eventos: list[Evento], frame: np.ndarray) -> list[dict]:
        nuevas = []
        ahora = time.time()
        for e in eventos:
            clave = f"{camara}:{e.clave}"
            with self._lock:
                if ahora - self._ultima.get(clave, 0) < self.enfriamiento:
                    continue
                self._ultima[clave] = ahora
            nuevas.append(self._guardar(camara, e, frame))
        return nuevas

    def _guardar(self, camara: str, e: Evento, frame: np.ndarray) -> dict:
        fecha = datetime.now()
        nombre = f"{fecha:%Y%m%d_%H%M%S_%f}_{camara}_{e.tipo}.jpg"
        img = frame.copy()
        for c in e.cajas:
            x1, y1, x2, y2 = (int(v) for v in c)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 230), 3)
        cv2.imwrite(str(self.carpeta / "capturas" / nombre), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO alertas (fecha, camara, tipo, severidad, mensaje, explicacion, captura) VALUES (?,?,?,?,?,?,?)",
                (fecha.isoformat(timespec="seconds"), camara, e.tipo, e.severidad, e.mensaje, e.explicacion, nombre))
            self._db.commit()
            aid = cur.lastrowid
        datos = {"id": aid, "fecha": fecha.isoformat(timespec="seconds"), "camara": camara, "tipo": e.tipo,
                 "tipo_nombre": NOMBRES.get(e.tipo, e.tipo), "severidad": e.severidad, "mensaje": e.mensaje,
                 "explicacion": e.explicacion, "captura": nombre, "atendida": 0, "veredicto": None}
        self._difundir({"evento": "alerta", "datos": datos})
        return datos

    def listar(self, limite: int = 100, tipo: str | None = None) -> list[dict]:
        sql = f"SELECT {', '.join(COLUMNAS)} FROM alertas"
        args: tuple = ()
        if tipo:
            sql += " WHERE tipo = ?"
            args = (tipo,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            filas = self._db.execute(sql, args + (limite,)).fetchall()
        out = [dict(zip(COLUMNAS, f)) for f in filas]
        for a in out:
            a["tipo_nombre"] = NOMBRES.get(a["tipo"], a["tipo"])
        return out

    def atender(self, aid: int) -> None:
        with self._lock:
            self._db.execute("UPDATE alertas SET atendida = 1 WHERE id = ?", (aid,))
            self._db.commit()
        self._difundir({"evento": "atendida", "datos": {"id": aid, "veredicto": None}})

    def dar_veredicto(self, aid: int, valor: str) -> None:
        """El operador revisa la alerta y la marca como confirmada o como falsa alarma."""
        if valor not in VEREDICTOS:
            raise ValueError(f"veredicto inválido: {valor}")
        with self._lock:
            self._db.execute("UPDATE alertas SET atendida = 1, veredicto = ? WHERE id = ?", (valor, aid))
            self._db.commit()
        self._difundir({"evento": "atendida", "datos": {"id": aid, "veredicto": valor}})

    def resumen(self) -> dict:
        with self._lock:
            filas = self._db.execute(
                "SELECT tipo, COUNT(*), SUM(atendida = 0) FROM alertas WHERE fecha >= date('now', 'localtime') GROUP BY tipo"
            ).fetchall()
        return {t: {"total": n, "pendientes": p or 0} for t, n, p in filas}

    def desempeno_operador(self) -> dict:
        """Precisión observada en esta instalación: confirmadas / (confirmadas + falsas), por tipo y total."""
        with self._lock:
            filas = self._db.execute(
                "SELECT tipo, SUM(veredicto = 'confirmada'), SUM(veredicto = 'falsa') FROM alertas GROUP BY tipo"
            ).fetchall()
        por_tipo = {}
        tc = tf = 0
        for t, c, f in filas:
            c, f = c or 0, f or 0
            tc, tf = tc + c, tf + f
            if c + f:
                por_tipo[t] = {"confirmadas": c, "falsas": f, "precision": round(c / (c + f), 3)}
        total = {"confirmadas": tc, "falsas": tf, "precision": round(tc / (tc + tf), 3) if tc + tf else None}
        return {"total": total, "por_tipo": por_tipo}

    def exportar_csv(self) -> str:
        with self._lock:
            filas = self._db.execute(f"SELECT {', '.join(COLUMNAS)} FROM alertas ORDER BY id").fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(COLUMNAS)
        w.writerows(filas)
        return buf.getvalue()

    def limpiar_antiguas(self, dias: int) -> None:
        limite = time.time() - dias * 86400
        for f in (self.carpeta / "capturas").glob("*.jpg"):
            if f.stat().st_mtime < limite:
                f.unlink(missing_ok=True)

    def cerrar(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def sse(datos: dict) -> str:
        return f"data: {json.dumps(datos, ensure_ascii=False)}\n\n"
