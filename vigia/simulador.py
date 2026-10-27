"""Modo demo: escena sintética con esqueletos que representan cada comportamiento.

Produce lo mismo que entregarían los modelos reales (personas con 17 puntos, IDs de seguimiento,
objetos y expresiones), así que las reglas, las alertas y el panel funcionan igual que con una
cámara. Sirve para probar el sistema y para exponer sin depender de la webcam.
Guion de 100 s que se repite:
  0-14 s   dos personas caminan (una alegre)
  14-40 s  merodeo en «Entrada» y luego intrusión en «Depósito»
  40-56 s  pelea entre dos personas
  56-70 s  caída y persona en el suelo
  70-100 s mochila abandonada y luego retirada por otra persona (posible hurto)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .tipos import EMOCIONES, ESQUELETO, Objeto, Observacion, Persona, Rostro

ANCHO, ALTO = 960, 540
DURACION = 100.0
H = 230.0  # altura de las personas en píxeles

ZONAS_DEMO = [
    {"id": "z1", "nombre": "Entrada", "tipo": "merodeo",
     "puntos": [[0.06, 0.56], [0.38, 0.56], [0.40, 0.80], [0.04, 0.80]]},
    {"id": "z2", "nombre": "Depósito", "tipo": "restringida",
     "puntos": [[0.74, 0.56], [0.97, 0.56], [0.98, 0.80], [0.72, 0.80]]},
]


def _emo(nombre: str) -> np.ndarray:
    p = np.full(len(EMOCIONES), 0.04)
    p[EMOCIONES.index(nombre)] = 1 - 0.04 * (len(EMOCIONES) - 1)
    return p


# Cada actor: id, emoción, [(t, x, y_pies, postura)], postura en {"camina", "quieto", "golpea", "tendido"}
ACTORES = [
    (1, "alegría", [(0, -0.05, 0.90, "camina"), (12, 1.05, 0.90, "camina")]),
    (2, "neutral", [(2, 1.05, 0.84, "camina"), (14, -0.05, 0.84, "camina")]),
    (3, "neutral", [(14, -0.05, 0.86, "camina"), (17, 0.20, 0.74, "camina"), (20, 0.32, 0.70, "camina"),
                    (23, 0.15, 0.76, "camina"), (26, 0.30, 0.72, "camina"), (29, 0.18, 0.75, "camina"),
                    (32, 0.28, 0.72, "camina"), (35, 0.84, 0.72, "camina"), (38, 0.86, 0.70, "quieto"),
                    (41, 1.08, 0.86, "camina")]),
    (4, "enojo", [(40, -0.05, 0.88, "camina"), (45, 0.42, 0.88, "golpea"), (51, 0.42, 0.88, "camina"),
                  (56, -0.05, 0.88, "camina")]),
    (5, "enojo", [(40, 1.05, 0.88, "camina"), (45, 0.57, 0.88, "golpea"), (51, 0.57, 0.88, "camina"),
                  (56, 1.05, 0.88, "camina")]),
    (6, "sorpresa", [(56, -0.05, 0.90, "camina"), (61, 0.50, 0.90, "cae"), (61.7, 0.50, 0.90, "tendido"),
                     (70, 0.50, 0.90, "tendido")]),
    (7, "neutral", [(70, -0.05, 0.90, "camina"), (74, 0.38, 0.90, "quieto"), (75, 0.38, 0.90, "camina"),
                    (79, -0.08, 0.90, "camina")]),
    (8, "neutral", [(86, 1.05, 0.90, "camina"), (90, 0.42, 0.90, "quieto"), (91, 0.42, 0.90, "camina"),
                    (97, 1.08, 0.90, "camina")]),
]


def _estado_actor(claves, t):
    if t < claves[0][0] or t > claves[-1][0]:
        return None
    for (t0, x0, y0, p0), (t1, x1, y1, _) in zip(claves, claves[1:]):
        if t0 <= t <= t1:
            a = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0), p0, a, (x1 - x0)
    t0, x0, y0, p0 = claves[-1]
    return x0, y0, p0, 1.0, 0.0


def esqueleto(cx: float, pies: float, postura: str, tiempo: float, avance: float,
              direccion: float, progreso: float) -> np.ndarray:
    """17 puntos COCO de una persona de altura H con los pies en (cx, pies)."""
    fase = 2 * math.pi * avance / (0.55 * H)
    swing = math.sin(fase) if postura == "camina" else 0.0
    d = 1 if direccion >= 0 else -1
    k = np.zeros((17, 2))
    alt = {  # (dx, altura sobre los pies) en fracción de H
        0: (0.02 * d, 0.93), 1: (0.035 * d, 0.95), 2: (0.0, 0.95), 3: (0.05 * d, 0.94), 4: (-0.02 * d, 0.94),
        5: (0.11, 0.80), 6: (-0.11, 0.80), 11: (0.07, 0.50), 12: (-0.07, 0.50),
    }
    for i, (dx, hy) in alt.items():
        k[i] = (cx + dx * H, pies - hy * H)
    # brazos
    for lado, (s, e, w) in enumerate([(5, 7, 9), (6, 8, 10)]):
        sgn = 1 if lado == 0 else -1
        osc = 0.10 * swing * sgn
        k[e] = (k[s][0] + (0.03 * sgn + osc * 0.6) * H, pies - 0.64 * H)
        k[w] = (k[s][0] + (0.03 * sgn + osc) * H, pies - 0.48 * H)
        if postura == "golpea":
            golpe = abs(math.sin(2 * math.pi * 1.7 * tiempo + lado * math.pi / 2))
            k[e] = (k[s][0] + d * 0.16 * H * (0.4 + golpe), pies - 0.76 * H)
            k[w] = (k[s][0] + d * 0.45 * H * (0.2 + golpe), pies - 0.80 * H)
    # piernas
    for lado, (hp, kn, an) in enumerate([(11, 13, 15), (12, 14, 16)]):
        sgn = 1 if lado == 0 else -1
        paso = 0.12 * swing * sgn
        k[kn] = (k[hp][0] + paso * 0.5 * H, pies - 0.27 * H)
        k[an] = (k[hp][0] + paso * H, pies)
    if postura in ("cae", "tendido"):
        # se rota el cuerpo alrededor de los pies hacia donde caminaba (con y hacia abajo,
        # un ángulo positivo gira en sentido horario: la cabeza termina a la derecha)
        phi = d * math.radians(88 * (min(1.0, progreso) if postura == "cae" else 1.0))
        piv = np.array([cx, pies])
        rel = k - piv
        c, s = math.cos(phi), math.sin(phi)
        k = piv + np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
        k[:, 1] = np.minimum(k[:, 1], pies)
    return k


def _caja(k: np.ndarray) -> np.ndarray:
    x1, y1 = k.min(axis=0)
    x2, y2 = k.max(axis=0)
    m = 0.06 * H
    return np.array([x1 - m, y1 - m * 1.3, x2 + m, y2 + 0.02 * H])


class Simulador:
    def __init__(self, semilla: int = 7):
        self.rng = np.random.default_rng(semilla)
        self.fondo = self._fondo()

    @staticmethod
    def _fondo() -> np.ndarray:
        img = np.zeros((ALTO, ANCHO, 3), np.uint8)
        img[:] = (46, 42, 38)
        img[int(ALTO * 0.55):] = (62, 60, 56)
        for i in range(-8, 9):
            cv2.line(img, (ANCHO // 2 + i * 40, int(ALTO * 0.55)), (ANCHO // 2 + i * 160, ALTO), (72, 70, 66), 1)
        for y in (0.62, 0.72, 0.85):
            cv2.line(img, (0, int(ALTO * y)), (ANCHO, int(ALTO * y)), (72, 70, 66), 1)
        cv2.rectangle(img, (int(ANCHO * 0.78), int(ALTO * 0.18)), (int(ANCHO * 0.94), int(ALTO * 0.55)), (70, 64, 58), -1)
        cv2.putText(img, "DEPOSITO", (int(ANCHO * 0.80), int(ALTO * 0.30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 115, 110), 1)
        cv2.rectangle(img, (int(ANCHO * 0.12), int(ALTO * 0.16)), (int(ANCHO * 0.30), int(ALTO * 0.55)), (64, 74, 82), -1)
        cv2.putText(img, "ENTRADA", (int(ANCHO * 0.15), int(ALTO * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 130, 140), 1)
        return img

    def frame(self, t_abs: float) -> tuple[np.ndarray, Observacion]:
        vuelta = int(t_abs // DURACION)
        t = t_abs % DURACION
        img = self.fondo.copy()
        obs = Observacion(t=t_abs)
        base = vuelta * 100
        manos: dict[int, np.ndarray] = {}
        for aid, emo, claves in ACTORES:
            e = _estado_actor(claves, t)
            if e is None:
                continue
            xn, yn, postura, prog, dx = e
            cx, pies = xn * ANCHO, yn * ALTO
            avance = abs(cx)
            k = esqueleto(cx, pies, postura, t, avance, dx if dx != 0 else 1, prog)
            k = k + self.rng.normal(0, 0.8, k.shape)       # ruido como el de un modelo real
            caja = _caja(k)
            if caja[2] < 0 or caja[0] > ANCHO:
                continue
            pid = base + aid
            obs.personas.append(Persona(id=pid, caja=caja, puntos=k, conf_puntos=np.full(17, 0.9)))
            manos[aid] = k[10] if dx < 0 else k[9]
            self._dibujar_persona(img, k, emo)
            r = 0.06 * H
            nariz = k[0]
            obs.rostros.append(Rostro(caja=np.array([nariz[0] - r, nariz[1] - r * 1.3, nariz[0] + r, nariz[1] + r]),
                                      probs=_emo(emo), persona_id=pid))
        # mochila: llevada por 7 (70-74 s), en el piso (74-90 s), llevada por 8 (90-97 s)
        mochila = None
        if 70 <= t < 74 and 7 in manos:
            mochila = manos[7] + np.array([0, 18])
        elif 74 <= t < 90:
            mochila = np.array([0.38 * ANCHO + 26, 0.90 * ALTO - 22])
        elif 90 <= t < 97 and 8 in manos:
            mochila = manos[8] + np.array([0, 18])
        if mochila is not None and 0 < mochila[0] < ANCHO:
            caja = np.array([mochila[0] - 18, mochila[1] - 22, mochila[0] + 18, mochila[1] + 22])
            obs.objetos.append(Objeto(id=base + 50, clase="mochila", caja=caja))
            x1, y1, x2, y2 = caja.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 90, 150), -1)
            cv2.rectangle(img, (x1 + 6, y1 + 8), (x2 - 6, y1 + 18), (30, 70, 120), -1)
        return img, obs

    @staticmethod
    def _dibujar_persona(img, k, emo):
        col = (205, 200, 190)
        for a, b in ESQUELETO[:12]:
            cv2.line(img, tuple(k[a].astype(int)), tuple(k[b].astype(int)), col, 9, cv2.LINE_AA)
        c = k[0].astype(int)
        r = int(0.06 * H)
        cv2.circle(img, (int(c[0]), int(c[1])), r, (150, 175, 205), -1, cv2.LINE_AA)
        boca = (int(c[0]), int(c[1] + r * 0.35))
        if emo == "alegría":
            cv2.ellipse(img, boca, (r // 2, r // 3), 0, 0, 180, (40, 40, 40), 2)
        elif emo in ("enojo", "tristeza"):
            cv2.ellipse(img, (boca[0], boca[1] + r // 4), (r // 2, r // 3), 0, 180, 360, (40, 40, 40), 2)
        elif emo == "sorpresa":
            cv2.circle(img, boca, r // 5, (40, 40, 40), 2)
        else:
            cv2.line(img, (boca[0] - r // 2, boca[1]), (boca[0] + r // 2, boca[1]), (40, 40, 40), 2)
        for dx in (-r // 3, r // 3):
            cv2.circle(img, (int(c[0] + dx), int(c[1] - r // 5)), 2, (40, 40, 40), -1)
