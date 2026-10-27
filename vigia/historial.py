"""Memoria temporal de cada persona y objeto seguido (por su ID de seguimiento)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tipos import Objeto, Observacion, Persona


@dataclass
class Muestra:
    t: float
    caja: np.ndarray
    puntos: Optional[np.ndarray]
    conf_puntos: Optional[np.ndarray]


@dataclass
class Pista:
    id: int
    primera_vez: float
    ultima_vez: float
    muestras: deque = field(default_factory=lambda: deque(maxlen=120))
    estado: dict = field(default_factory=dict)   # memoria propia de cada regla

    def ventana(self, t_ahora: float, segundos: float) -> list[Muestra]:
        return [m for m in self.muestras if t_ahora - m.t <= segundos]

    def alto_de_pie(self, t_ahora: float, segundos: float = 4.0) -> float:
        """Altura más alta reciente: sirve de escala aunque la persona esté agachada o tendida."""
        v = self.ventana(t_ahora, segundos) or list(self.muestras)
        return float(max(m.caja[3] - m.caja[1] for m in v)) if v else 1.0


class Historial:
    def __init__(self, olvido: float = 5.0):
        self.personas: dict[int, Pista] = {}
        self.objetos: dict[int, Pista] = {}
        self.clase_objeto: dict[int, str] = {}
        self.olvido = olvido

    def actualizar(self, obs: Observacion) -> None:
        for p in obs.personas:
            self._agregar(self.personas, p.id, obs.t, p.caja, p.puntos, p.conf_puntos)
        for o in obs.objetos:
            self._agregar(self.objetos, o.id, obs.t, o.caja, None, None)
            self.clase_objeto[o.id] = o.clase
        for tabla in (self.personas, self.objetos):
            for k in [k for k, v in tabla.items() if obs.t - v.ultima_vez > self.olvido]:
                del tabla[k]

    @staticmethod
    def _agregar(tabla, id_, t, caja, puntos, conf):
        pista = tabla.get(id_)
        if pista is None:
            pista = tabla[id_] = Pista(id=id_, primera_vez=t, ultima_vez=t)
        pista.ultima_vez = t
        pista.muestras.append(Muestra(t, np.asarray(caja, float),
                                      None if puntos is None else np.asarray(puntos, float),
                                      None if conf is None else np.asarray(conf, float)))

    def visibles(self, t: float, tabla: str = "personas") -> list[Pista]:
        d = self.personas if tabla == "personas" else self.objetos
        return [p for p in d.values() if t - p.ultima_vez < 0.5]
