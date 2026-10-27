"""Lectura de cámaras (webcam, RTSP o archivo) y análisis en un hilo por cámara."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

from .alertas import GestorAlertas
from .dibujo import cajas_cabeza, difuminar, dibujar
from .historial import Historial
from .reglas import Reglas
from .tipos import Observacion

log = logging.getLogger("vigia.camara")


class Lector:
    """Lee frames en su propio hilo y conserva solo el último (evita acumular retraso en RTSP)."""

    def __init__(self, fuente):
        self.fuente = int(fuente) if str(fuente).isdigit() else str(fuente)
        self.es_archivo = isinstance(self.fuente, str) and not self.fuente.lower().startswith(("rtsp", "http"))
        self.frame: Optional[np.ndarray] = None
        self.conectada = False
        self._parar = False
        self._lock = threading.Lock()
        threading.Thread(target=self._bucle, daemon=True).start()

    def _abrir(self):
        if isinstance(self.fuente, int):
            cap = cv2.VideoCapture(self.fuente, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.fuente)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.fuente)
        else:
            cap = cv2.VideoCapture(self.fuente)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _bucle(self):
        espera = 1.0
        while not self._parar:
            cap = self._abrir()
            if not cap.isOpened():
                self.conectada = False
                log.warning("No se pudo abrir %s; reintento en %.0f s", self.fuente, espera)
                time.sleep(espera)
                espera = min(espera * 2, 30)
                continue
            espera = 1.0
            fps_archivo = cap.get(cv2.CAP_PROP_FPS) or 25
            while not self._parar:
                ok, f = cap.read()
                if not ok:
                    if self.es_archivo:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                with self._lock:
                    self.frame = f
                self.conectada = True
                if self.es_archivo:
                    time.sleep(1.0 / fps_archivo)
            cap.release()
            self.conectada = False

    def ultimo(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def detener(self):
        self._parar = True


class Trabajador:
    def __init__(self, cam: dict, cfg: dict, alertas: GestorAlertas, zonas: list[dict], demo: bool = False):
        self.id = cam["id"]
        self.nombre = cam.get("nombre", cam["id"])
        self.fuente = cam["fuente"]
        self.cfg = cfg
        self.alertas = alertas
        self.zonas = zonas
        self.demo = demo or str(self.fuente) == "demo"
        self.hist = Historial()
        self.reglas = Reglas(cfg["reglas"])
        self.jpeg: Optional[bytes] = None
        self.crudo: Optional[np.ndarray] = None   # último frame sin anotaciones (ya difuminado)
        self.estado = {"id": self.id, "nombre": self.nombre, "conectada": False, "fps": 0.0, "personas": 0,
                       "emociones": {}, "fuente": str(self.fuente), "error": None}
        self._en_alerta: dict[int, float] = {}
        self._tiempos: deque = deque(maxlen=20)
        self._parar = False
        self._t0 = time.time()
        self.lector = None
        self.sim = None
        self.detector = None
        self.rostros = None
        threading.Thread(target=self._iniciar, daemon=True).start()

    # ---- arranque (la carga de modelos puede tardar la primera vez)
    def _iniciar(self):
        try:
            if self.demo:
                from .simulador import Simulador
                self.sim = Simulador()
            else:
                from .modelos import AnalizadorRostros, DetectorYOLO
                self.lector = Lector(self.fuente)
                self.detector = DetectorYOLO(self.cfg["modelos"])
                self.rostros = AnalizadorRostros(self.cfg["modelos"]["rostros"], self.cfg["modelos"]["emociones"])
        except Exception as e:  # p. ej. falta instalar ultralytics
            log.exception("No se pudo iniciar la cámara %s", self.id)
            self.estado["error"] = str(e)
            return
        self._bucle()

    def _bucle(self):
        periodo = 1.0 / self.cfg["procesamiento"]["fps_objetivo"]
        while not self._parar:
            inicio = time.time()
            try:
                self._paso()
            except Exception as e:
                log.exception("Error procesando %s", self.id)
                self.estado["error"] = str(e)
                time.sleep(1)
            time.sleep(max(0.0, periodo - (time.time() - inicio)))

    def _paso(self):
        priv = self.cfg["privacidad"]
        if self.demo:
            t = time.time() - self._t0
            frame, obs = self.sim.frame(t)
            self.estado["conectada"] = True
            if not priv["analizar_emociones"]:
                for r in obs.rostros:
                    r.probs = None
        else:
            frame = self.lector.ultimo()
            self.estado["conectada"] = self.lector.conectada
            if frame is None:
                return
            ancho_max = self.cfg["procesamiento"]["ancho_max"]
            if frame.shape[1] > ancho_max:
                esc = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, (ancho_max, int(frame.shape[0] * esc)))
            t = time.time()
            obs = self.detector.procesar(frame, t)
            if self.rostros.disponible and (priv["difuminar_rostros"] or priv["analizar_emociones"]):
                from .modelos import asignar_rostros
                obs.rostros = self.rostros.detectar(frame, priv["analizar_emociones"])
                asignar_rostros(obs.rostros, obs.personas)

        h, w = frame.shape[:2]
        self.hist.actualizar(obs)
        eventos = self.reglas.evaluar(obs, self.hist, w, h, self.zonas)
        ahora = time.time()
        for e in eventos:
            for c in e.cajas:
                for p in obs.personas:
                    if np.allclose(p.caja, c):
                        self._en_alerta[p.id] = ahora
        en_alerta = {pid for pid, ta in self._en_alerta.items() if ahora - ta < 4}

        emociones = {}
        if priv["analizar_emociones"]:
            for p in obs.personas:
                est = self.hist.personas.get(p.id)
                if est and "emo_etiqueta" in est.estado:
                    emociones[p.id] = est.estado["emo_etiqueta"]

        if priv["difuminar_rostros"]:
            difuminar(frame, cajas_cabeza(obs))
        self.crudo = frame.copy()
        self._tiempos.append(ahora)
        fps = (len(self._tiempos) - 1) / (self._tiempos[-1] - self._tiempos[0]) if len(self._tiempos) > 1 else 0.0
        enc = f"{self.nombre}  |  {time.strftime('%H:%M:%S')}  |  {fps:.1f} fps  |  {len(obs.personas)} personas"
        if self.demo:
            enc += "  |  MODO DEMO (escena simulada)"
        dibujar(frame, obs, self.zonas, en_alerta, emociones, enc)
        self.alertas.procesar(self.id, eventos, frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
        self.estado.update(fps=round(fps, 1), personas=len(obs.personas),
                           emociones=dict(Counter(emociones.values())), error=None, ancho=w, alto=h)

    def actualizar_cfg(self, cfg: dict):
        self.cfg = cfg
        self.reglas.actualizar_cfg(cfg["reglas"])

    def detener(self):
        self._parar = True
        if self.lector:
            self.lector.detener()
