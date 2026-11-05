"""Estructuras de datos compartidas por los detectores, las reglas y el panel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Índices de los 17 puntos COCO que entrega YOLO-pose
NARIZ, OJO_I, OJO_D, OREJA_I, OREJA_D = 0, 1, 2, 3, 4
HOMBRO_I, HOMBRO_D, CODO_I, CODO_D, MUNECA_I, MUNECA_D = 5, 6, 7, 8, 9, 10
CADERA_I, CADERA_D, RODILLA_I, RODILLA_D, TOBILLO_I, TOBILLO_D = 11, 12, 13, 14, 15, 16

ESQUELETO = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

EMOCIONES = ["enojo", "desagrado", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]


# Proporciones del cuerpo respecto de la estatura (tablas antropométricas de Drillis y Contini)
_SEGMENTOS = [((5, 6), (11, 12), 0.29),   # hombros-caderas (tronco)
              ((11,), (13,), 0.245), ((12,), (14,), 0.245),   # muslos
              ((13,), (15,), 0.246), ((14,), (16,), 0.246),   # canillas
              ((5,), (7,), 0.186), ((6,), (8,), 0.186),       # brazos
              ((5,), (6,), 0.259)]                          # ancho de hombros


def alto_estimado(caja, puntos=None, conf=None, conf_min: float = 0.3) -> float:
    """Estatura en píxeles. Si la cámara corta a la persona (solo de la cintura para arriba o para abajo),
    la caja mide la mitad; las partes del cuerpo que sí se ven permiten estimar la estatura completa."""
    h = float(max(1.0, caja[3] - caja[1]))
    if puntos is None:
        return h
    ok = lambda i: conf is None or conf[i] >= conf_min
    est = []
    for a, b, frac in _SEGMENTOS:
        if all(ok(i) for i in a + b):
            pa = np.mean([puntos[i] for i in a], axis=0)
            pb = np.mean([puntos[i] for i in b], axis=0)
            est.append(float(np.linalg.norm(pa - pb)) / frac)
    if not est:
        return h
    est.sort(reverse=True)
    e = est[1] if len(est) > 1 else est[0]      # la segunda mayor: una parte mal detectada no la infla
    return max(h, min(e, 2.5 * h))              # vista de lado o en escorzo, las partes solo se acortan


@dataclass
class Persona:
    id: int
    caja: np.ndarray                      # x1, y1, x2, y2 en píxeles
    puntos: Optional[np.ndarray] = None   # (17, 2) en píxeles
    conf_puntos: Optional[np.ndarray] = None  # (17,)
    conf: float = 1.0
    _alto: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    @property
    def alto(self) -> float:
        """Altura de la persona entera, aunque la cámara solo vea medio cuerpo (ver alto_estimado)."""
        if self._alto is None:
            self._alto = alto_estimado(self.caja, self.puntos, self.conf_puntos)
        return self._alto

    @property
    def centro(self) -> np.ndarray:
        return np.array([(self.caja[0] + self.caja[2]) / 2, (self.caja[1] + self.caja[3]) / 2])

    @property
    def pies(self) -> np.ndarray:
        return np.array([(self.caja[0] + self.caja[2]) / 2, self.caja[3]])

    def punto(self, i: int, conf_min: float = 0.3) -> Optional[np.ndarray]:
        if self.puntos is None:
            return None
        if self.conf_puntos is not None and self.conf_puntos[i] < conf_min:
            return None
        return self.puntos[i]


@dataclass
class Objeto:
    id: int
    clase: str          # "mochila", "bolso", "maleta"
    caja: np.ndarray
    conf: float = 1.0

    @property
    def centro(self) -> np.ndarray:
        return np.array([(self.caja[0] + self.caja[2]) / 2, (self.caja[1] + self.caja[3]) / 2])


@dataclass
class Rostro:
    caja: np.ndarray                       # x1, y1, x2, y2
    probs: Optional[np.ndarray] = None     # probabilidades de EMOCIONES
    persona_id: Optional[int] = None


@dataclass
class Observacion:
    """Todo lo que los modelos ven en un frame."""
    t: float
    personas: list[Persona] = field(default_factory=list)
    objetos: list[Objeto] = field(default_factory=list)
    rostros: list[Rostro] = field(default_factory=list)


@dataclass
class Evento:
    """Lo que una regla reporta; el gestor de alertas decide si se convierte en alerta."""
    tipo: str            # violencia, caida, en_suelo, merodeo, intrusion, abandono, hurto, emocion
    severidad: str       # alta, media, baja
    mensaje: str
    explicacion: str     # evidencia numérica que justifica la alerta (IA explicable)
    clave: str           # para evitar duplicados (p. ej. "violencia:3-7")
    cajas: list = field(default_factory=list)  # cajas a resaltar en la captura
