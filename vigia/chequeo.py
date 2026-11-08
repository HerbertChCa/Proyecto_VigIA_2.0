"""Chequeo previo a una demostración: ¿está todo listo para que VigIA detecte bien en esta sala?

La mayoría de los fallos en una prueba en vivo no son de las reglas sino del lugar: poca luz o una ventana detrás
de las personas, una cámara que se mueve, personas muy lejos o con los pies fuera de la imagen, una PC que no
alcanza los cuadros por segundo, una mochila que el detector no ve. El chequeo mide esas cosas con lo que la cámara
está viendo en los últimos 30 s y dice qué corregir antes de que llegue el público (botón «Chequeo» del panel).

Cada punto queda en uno de cuatro estados: ok, aviso (funciona peor), mal (corregir antes de empezar) y
pendiente (falta hacer una prueba frente a la cámara, por ejemplo mostrar la mochila).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .reglas import PORTATILES

TOBILLO_I, TOBILLO_D = 15, 16


class Sondeo:
    """Junta, cuadro a cuadro y sin costo apreciable, las medidas que usa el chequeo (últimos `ventana` s)."""

    def __init__(self, ventana: float = 30.0):
        self.ventana = ventana
        self.personas: deque = deque()     # (t, id, alto relativo, pies visibles)
        self.cuenta: deque = deque()       # (t, personas a la vez)
        self.objetos: deque = deque()      # (t, clase)
        self.luz: deque = deque()          # (t, brillo medio, brillo de las personas o None)
        self.movimiento: deque = deque()   # (t, desplazamiento de la imagen en px)
        self._previa: Optional[np.ndarray] = None
        self._t_imagen = -1e9

    def _recortar(self, t: float) -> None:
        for d in (self.personas, self.cuenta, self.objetos, self.luz, self.movimiento):
            while d and t - d[0][0] > self.ventana:
                d.popleft()

    def actualizar(self, frame: np.ndarray, obs, t: Optional[float] = None) -> None:
        t = time.time() if t is None else t
        h = frame.shape[0]
        self.cuenta.append((t, len(obs.personas)))
        for p in obs.personas:
            pies = (p.conf_puntos is not None and len(p.conf_puntos) > TOBILLO_D
                    and min(p.conf_puntos[TOBILLO_I], p.conf_puntos[TOBILLO_D]) >= 0.3)
            self.personas.append((t, p.id, float(p.caja[3] - p.caja[1]) / h, bool(pies)))
        for o in obs.objetos:
            if o.clase in PORTATILES:
                self.objetos.append((t, o.clase))
        if t - self._t_imagen >= 1.0:          # la luz y el movimiento de cámara, una vez por segundo
            self._t_imagen = t
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dentro = []
            for p in obs.personas:
                x1, y1, x2, y2 = (int(max(0, v)) for v in p.caja)
                if x2 - x1 > 4 and y2 - y1 > 4:
                    dentro.append(float(gris[y1:y2, x1:x2].mean()))
            self.luz.append((t, float(gris.mean()), float(np.mean(dentro)) if dentro else None))
            # movimiento de la cámara = desplazamiento del fondo: las personas se tapan (si no, quien camina
            # parece una cámara que se mueve) y una medida poco confiable (fondo liso) no se cuenta
            esc = 320 / frame.shape[1]
            chica = np.float32(cv2.resize(gris, (320, int(h * esc))))
            fondo = float(np.median(chica))
            for p in obs.personas:
                x1, y1, x2, y2 = (int(max(0, v * esc)) for v in p.caja)
                chica[y1:y2 + 1, x1:x2 + 1] = fondo
            if self._previa is not None and self._previa.shape == chica.shape:
                ventana = cv2.createHanningWindow(chica.shape[::-1], cv2.CV_32F)
                (dx, dy), confianza = cv2.phaseCorrelate(self._previa, chica, ventana)
                if confianza >= 0.15:
                    self.movimiento.append((t, float(np.hypot(dx, dy)) / esc))
            self._previa = chica
        self._recortar(t)


def _punto(nombre, estado, detalle, consejo=""):
    return {"nombre": nombre, "estado": estado, "detalle": detalle, "consejo": consejo}


def evaluar(s: Sondeo, fps: float, revision: bool, contexto: dict) -> dict:
    """Devuelve {"listo": bool, "puntos": [...]}. `contexto`: zonas, aceleracion, vocabulario, emociones, video."""
    puntos = []

    # 1. velocidad: cuántos cuadros por segundo de acción se analizan
    if contexto.get("video_detallado"):
        puntos.append(_punto("Velocidad del análisis", "ok", "Video en modo detallado: se analiza cada segundo completo."))
    elif fps >= 7:
        puntos.append(_punto("Velocidad del análisis", "ok", f"{fps:.1f} cuadros por segundo: se ven golpes y tirones."))
    elif revision:
        puntos.append(_punto("Velocidad del análisis", "aviso",
                             f"{fps:.1f} cuadros por segundo. La segunda mirada revisa cuadro a cuadro los incidentes "
                             "rápidos: sus alertas llegan unos segundos tarde.",
                             "Para que lleguen antes: python -m vigia --rapido y scripts\\acelerar_modelos.py; "
                             "deje una sola cámara."))
    else:
        puntos.append(_punto("Velocidad del análisis", "mal" if fps < 4 else "aviso",
                             f"{fps:.1f} cuadros por segundo: los tirones y golpes pueden caer entre dos cuadros.",
                             "python -m vigia --rapido, scripts\\acelerar_modelos.py, una sola cámara."))

    # 2. luz
    if not s.luz:
        puntos.append(_punto("Luz", "pendiente", "Todavía no hay imagen."))
    else:
        brillo = float(np.median([b for _, b, _ in s.luz]))
        personas = [bp for _, _, bp in s.luz if bp is not None]
        if brillo < 60:
            puntos.append(_punto("Luz", "mal", f"Imagen oscura (brillo {brillo:.0f} de 255): el esqueleto se pierde.",
                                 "Prenda las luces o acerque una lámpara; evite que la cámara apunte a zonas oscuras."))
        elif brillo > 210:
            puntos.append(_punto("Luz", "aviso", f"Imagen muy clara (brillo {brillo:.0f} de 255).",
                                 "Baje la exposición del celular o aleje la cámara de la luz directa."))
        elif personas and float(np.median(personas)) < brillo - 45:
            puntos.append(_punto("Luz", "aviso",
                                 f"Contraluz: las personas se ven más oscuras ({np.median(personas):.0f}) que el fondo "
                                 f"({brillo:.0f}). Suele haber una ventana o una pantalla detrás.",
                                 "Gire la cámara para que la luz venga de detrás de ella, no de detrás de las personas."))
        else:
            puntos.append(_punto("Luz", "ok", f"Brillo {brillo:.0f} de 255, sin contraluz."))

    # 3. distancia y 4. cuerpo completo
    if not s.personas:
        puntos.append(_punto("Distancia", "pendiente", "No se vio a nadie en los últimos 30 s.",
                             "Párese frente a la cámara, donde se harán las pruebas, unos segundos."))
    else:
        alto = float(np.median([a for _, _, a, _ in s.personas]))
        if alto < 0.25:
            puntos.append(_punto("Distancia", "aviso", f"Las personas ocupan {100 * alto:.0f} % de la altura de la "
                                 "imagen: de lejos el esqueleto y los objetos se detectan peor.",
                                 "Acerque la cámara o las pruebas: que la persona ocupe al menos un tercio de la altura."))
        elif alto > 0.92:
            puntos.append(_punto("Distancia", "aviso", "Las personas llenan la imagen: se cortan la cabeza o los pies.",
                                 "Aleje la cámara un poco."))
        else:
            puntos.append(_punto("Distancia", "ok", f"Las personas ocupan {100 * alto:.0f} % de la altura de la imagen."))
        pies = float(np.mean([p for _, _, _, p in s.personas]))
        puntos.append(_punto("Cuerpo completo", "ok" if pies >= 0.6 else "aviso",
                             f"Se ven los pies en {100 * pies:.0f} % de los cuadros.",
                             "" if pies >= 0.6 else "Que se vea el cuerpo entero, del piso a la cabeza: sin pies, las "
                             "caídas y las personas en el suelo se detectan peor."))

    # 5. cámara quieta
    if s.movimiento:
        mov = float(np.median([m for _, m in s.movimiento]))
        puntos.append(_punto("Cámara quieta", "ok" if mov <= 2 else "mal",
                             f"La imagen se desplaza {mov:.1f} px por segundo.",
                             "" if mov <= 2 else "Fije la cámara (trípode o apoyo). Si se mueve, las zonas dejan de "
                             "coincidir con el piso y las velocidades de las personas dejan de ser reales."))

    # 6. seguimiento estable
    if s.cuenta and s.personas:
        ids = {pid for _, pid, _, _ in s.personas}
        a_la_vez = max(1, max(n for _, n in s.cuenta))
        estable = len(ids) <= 3 * a_la_vez + 2
        puntos.append(_punto("Seguimiento", "ok" if estable else "aviso",
                             f"{len(ids)} números de seguimiento para {a_la_vez} persona(s) a la vez.",
                             "" if estable else "Las personas pierden su número (se tapan, salen y entran, o la "
                             "cámara está muy cerca): las reglas pierden su historia."))

    # 7. objetos
    clases = sorted({c for _, c in s.objetos})
    if not contexto.get("vocabulario"):
        puntos.append(_punto("Detector de objetos", "aviso", "Solo las 80 clases de COCO: no reconoce ropa, bolsas ni sacos.",
                             "pip install git+https://github.com/ultralytics/CLIP.git y python scripts\\preparar_vocabulario.py"))
    puntos.append(_punto("Objetos", "ok" if clases else "pendiente",
                         f"Vistos en los últimos 30 s: {', '.join(clases)}." if clases else "No se vio ninguna mochila, "
                         "bolso ni prenda.", "" if clases else "Muestre a la cámara la mochila que usarán, en la mano y "
                         "a la espalda, unos segundos: debe aparecer con su recuadro."))

    # 8. zonas, modelos y emociones
    zonas = contexto.get("zonas") or []
    tipos = {z.get("tipo") for z in zonas}
    falta = [n for n, t in (("restringida (intrusión)", "restringida"), ("de merodeo", "merodeo"),
                            ("protegida (vandalismo)", "protegida")) if t not in tipos]
    puntos.append(_punto("Zonas", "ok" if not falta else "aviso",
                         f"{len(zonas)} zona(s) dibujada(s)." + (f" Falta una zona {' y una '.join(falta)}." if falta else ""),
                         "" if not falta else "Botón «Zonas»: dibuje sobre el piso las zonas de los retos."))
    acel = contexto.get("aceleracion") or {}
    if acel:
        lentos = [k for k, v in acel.items() if v == "cpu"]
        puntos.append(_punto("Modelos", "ok" if not lentos else "aviso",
                             ", ".join(f"{k} en {v}" for k, v in acel.items()) + ".",
                             "" if not lentos else "python scripts\\acelerar_modelos.py (2-3 veces más rápido en CPU)."))
    if not contexto.get("emociones"):
        puntos.append(_punto("Expresiones", "aviso", "El análisis de expresiones está apagado: el reto «Sonríe» no "
                             "funcionará.", "Ajustes > Analizar expresiones faciales (solo si van a usar ese reto)."))

    listo = not any(p["estado"] in ("mal", "pendiente") for p in puntos)
    return {"listo": listo, "puntos": puntos}
