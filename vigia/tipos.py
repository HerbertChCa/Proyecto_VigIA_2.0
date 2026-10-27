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


@dataclass
class Persona:
    id: int
    caja: np.ndarray                      # x1, y1, x2, y2 en píxeles
    puntos: Optional[np.ndarray] = None   # (17, 2) en píxeles
    conf_puntos: Optional[np.ndarray] = None  # (17,)
    conf: float = 1.0

    @property
    def alto(self) -> float:
        return float(max(1.0, self.caja[3] - self.caja[1]))

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
