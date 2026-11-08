"""Segunda mirada: revisión cuadro a cuadro de los últimos segundos de una cámara en vivo.

Una cámara en vivo no puede esperar al análisis. Si la computadora alcanza 2-3 cuadros por segundo, un tirón de
mochila o un forcejeo de 2 s caen entre pocos cuadros y las reglas no llegan a verlos. Un video subido no tiene
ese problema: se analiza en «cámara lenta», 10 cuadros por cada segundo de video.

La segunda mirada junta las dos cosas. Mientras la vista en vivo sigue a su ritmo:

1. El lector guarda en memoria los últimos segundos de la cámara, 10 cuadros por segundo, en JPEG (unos 60 kB
   cada uno: 15 s ocupan menos de 10 MB).
2. Cuando la vista en vivo nota algo que podría ser un incidente rápido (dos personas juntas, un movimiento
   brusco, alguien que queda horizontal), pide revisar un tramo: unos segundos antes y unos después.
3. Un segundo analizador, con sus propios modelos, seguimiento, historial y reglas, recorre ese tramo cuadro a
   cuadro, igual que un video en modo detallado.
4. Lo que encuentre y la vista en vivo no haya avisado se convierte en alerta, con una explicación que dice que
   salió de la revisión. La alerta llega unos segundos tarde, pero llega.

Las reglas de situaciones lentas (merodeo, objeto abandonado, intrusión, aglomeración) no se revisan: la vista en
vivo las ve bien aunque analice pocos cuadros, porque duran decenas de segundos.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np

from .historial import Historial
from .objetos import FiltroObjetos
from .reglas import PORTATILES, Reglas, alto_estimado, cuerpos_juntos, sostiene

log = logging.getLogger("vigia.revision")

# situaciones rápidas: las que la vista en vivo puede perder entre dos cuadros
REVISABLES = {"violencia", "caida", "en_suelo", "hurto", "robo_huida", "arma"}

REVISION = {
    "activa": True,
    "solo_si_fps_menor_a": 8,     # con 8 o más cuadros por segundo en vivo, la segunda mirada no aporta
    "antes": 4.0,                 # segundos antes de la sospecha que se revisan
    "despues": 3.0,               # y después (el incidente puede seguir)
    "memoria": 15.0,              # segundos que se guardan
    "fps": 10,                    # cuadros guardados por segundo
    "distancia_sospecha": 1.3,    # alturas entre dos personas para sospechar
    "pausa_entre_sospechas": 2.0,
}


class Memoria:
    """Los últimos segundos de una cámara, comprimidos en JPEG. La escribe el hilo del lector y la lee el revisor."""

    def __init__(self, segundos: float = 15.0, fps: float = 10, ancho: int = 960, calidad: int = 85):
        self.fps = max(1.0, float(fps))
        self.ancho = ancho
        self.calidad = calidad
        self._cuadros: deque = deque(maxlen=max(1, int(segundos * self.fps)))
        self._ultimo = -1e9
        self._lock = threading.Lock()

    def guardar(self, frame: np.ndarray, t: float) -> bool:
        """Guarda el cuadro si ya pasó 1/fps desde el anterior. Devuelve si lo guardó."""
        if t - self._ultimo < 0.95 / self.fps:
            return False
        if frame.shape[1] > self.ancho:
            e = self.ancho / frame.shape[1]
            frame = cv2.resize(frame, (self.ancho, int(frame.shape[0] * e)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.calidad])
        if not ok:
            return False
        with self._lock:
            self._cuadros.append((t, buf))
            self._ultimo = t
        return True

    def entre(self, desde: float, hasta: float) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return [(t, b) for t, b in self._cuadros if desde <= t <= hasta]

    @property
    def desde(self) -> Optional[float]:
        with self._lock:
            return self._cuadros[0][0] if self._cuadros else None

    def __len__(self) -> int:
        return len(self._cuadros)


def sospecha(obs, reglas: Reglas, cfg: dict) -> Optional[str]:
    """¿Hay algo en este cuadro que podría ser un incidente rápido? Devuelve el motivo o None. Es barato: usa lo
    que la vista en vivo ya calculó."""
    personas = obs.personas
    for i, a in enumerate(personas):
        for b in personas[i + 1:]:
            ca = np.array([(a.caja[0] + a.caja[2]) / 2, (a.caja[1] + a.caja[3]) / 2])
            cb = np.array([(b.caja[0] + b.caja[2]) / 2, (b.caja[1] + b.caja[3]) / 2])
            alto = (alto_estimado(a.caja, a.puntos, a.conf_puntos) + alto_estimado(b.caja, b.puntos, b.conf_puntos)) / 2
            if cuerpos_juntos(a, b) or float(np.linalg.norm(ca - cb)) / max(alto, 1.0) <= cfg["distancia_sospecha"]:
                return f"personas #{a.id} y #{b.id} juntas"
    umbral = reglas.cfg.get("violencia", {}).get("velocidad_extremidades", 1.8)
    for pid, (_, v) in getattr(reglas, "_rap", {}).items():
        if v >= 0.6 * umbral:
            return f"movimiento brusco de la persona #{pid}"
    for p in personas:
        w, h = p.caja[2] - p.caja[0], p.caja[3] - p.caja[1]
        if w > 1.1 * h:
            return f"persona #{p.id} horizontal"
    if len(personas) >= 2:
        for o in obs.objetos:
            if o.clase in PORTATILES and any(sostiene(p, o.caja, clase=o.clase) for p in personas):
                return f"{o.clase} cerca de otra persona"
    return None


class Revisor:
    """Revisa tramos de la memoria de una cámara con un analizador propio, en su propio hilo."""

    def __init__(self, cfg: dict, memoria: Memoria, crear_detector: Callable, emitir: Callable,
                 ignorar: Callable[[], list] = lambda: []):
        self.cfg = cfg                      # configuración completa (modelos, reglas, procesamiento)
        self.memoria = memoria
        self._crear_detector = crear_detector
        self._emitir = emitir               # emitir(evento, imagen, obs, tramo)
        self._ignorar = ignorar
        self.detector = None
        self._pendientes: list[dict] = []
        self._revisado_hasta = -1e9
        self._lock = threading.Lock()
        self._parar = False
        self.estado = {"revisando": False, "progreso": 0, "pendientes": 0, "tramos": 0, "hallazgos": 0,
                       "ultimo": None}
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    # ---- pedidos
    def pedir(self, desde: float, hasta: float, motivo: str, forzar: bool = False) -> None:
        """Pide revisar [desde, hasta]. Se une con un pedido pendiente que se superponga y no repite lo ya revisado
        (salvo `forzar`: el operador lo pidió a mano)."""
        if not forzar:
            desde = max(desde, self._revisado_hasta)
        if hasta - desde < 0.5:
            return
        with self._lock:
            for p in self._pendientes:
                if desde <= p["hasta"] + 1.0 and hasta >= p["desde"] - 1.0:
                    p["desde"], p["hasta"] = min(p["desde"], desde), max(p["hasta"], hasta)
                    if motivo not in p["motivos"]:
                        p["motivos"].append(motivo)
                    break
            else:
                self._pendientes.append({"desde": desde, "hasta": hasta, "motivos": [motivo]})
            self.estado["pendientes"] = len(self._pendientes)

    def detener(self) -> None:
        self._parar = True

    # ---- hilo
    def _bucle(self) -> None:
        while not self._parar:
            tramo = None
            with self._lock:
                if self._pendientes and self._pendientes[0]["hasta"] <= time.time():
                    tramo = self._pendientes.pop(0)
                self.estado["pendientes"] = len(self._pendientes)
            if tramo is None:
                time.sleep(0.1)
                continue
            try:
                self.revisar(tramo)
            except Exception:  # noqa: BLE001 - un tramo con error no detiene la cámara
                log.exception("Error en la revisión de un tramo")
            finally:
                self.estado.update(revisando=False, progreso=0)

    def revisar(self, tramo: dict) -> list:
        """Recorre el tramo cuadro a cuadro; devuelve los eventos emitidos."""
        cuadros = self.memoria.entre(tramo["desde"], tramo["hasta"])
        self._revisado_hasta = max(self._revisado_hasta, tramo["hasta"])
        if len(cuadros) < 3:
            return []
        if self.detector is None:
            self.detector = self._crear_detector()
        if hasattr(self.detector, "reiniciar_seguimiento"):
            self.detector.reiniciar_seguimiento()
        self.detector.cada_n = 2
        reglas_cfg = dict(self.cfg["reglas"])
        reglas_cfg["objetos"] = {**reglas_cfg.get("objetos", {}), "calentamiento": 0}   # el tramo empieza a mitad
        hist, filtro, reglas = Historial(), FiltroObjetos(abierto=getattr(self.detector, "abierto", False)), Reglas(reglas_cfg)
        reglas.ignorar = list(self._ignorar())
        self.estado.update(revisando=True, progreso=0, tramos=self.estado["tramos"] + 1)
        emitidos, tipos = [], set()
        t_ini = time.time()
        for k, (t, buf) in enumerate(cuadros):
            if self._parar and threading.current_thread() is self._hilo:
                break                                   # la cámara se quitó a mitad de una revisión
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            obs = self.detector.procesar(frame, t)
            obs.objetos = filtro.filtrar(obs.objetos, t, reglas_cfg["objetos"], obs.personas)
            hist.actualizar(obs)
            for e in reglas.evaluar(obs, hist, frame.shape[1], frame.shape[0], []):
                if e.tipo in REVISABLES and e.tipo not in tipos:
                    tipos.add(e.tipo)
                    if self._emitir(e, frame, obs, tramo):
                        emitidos.append(e)
            self.estado["progreso"] = round(100 * (k + 1) / len(cuadros))
        self.estado["hallazgos"] += len(emitidos)
        self.estado["ultimo"] = {"segundos": round(tramo["hasta"] - tramo["desde"], 1), "cuadros": len(cuadros),
                                 "tardo": round(time.time() - t_ini, 1), "motivos": tramo["motivos"],
                                 "hallazgos": [e.tipo for e in emitidos]}
        log.info("Revisión de %.1f s (%d cuadros, %s): %s", tramo["hasta"] - tramo["desde"], len(cuadros),
                 ", ".join(tramo["motivos"]), ", ".join(e.tipo for e in emitidos) or "nada nuevo")
        return emitidos
