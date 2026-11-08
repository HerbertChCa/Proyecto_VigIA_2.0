"""Memoria temporal de cada persona y objeto seguido (por su ID de seguimiento)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tipos import Objeto, Observacion, Persona, alto_estimado


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Muestra:
    t: float
    caja: np.ndarray
    puntos: Optional[np.ndarray]
    conf_puntos: Optional[np.ndarray]
    alto: float = 0.0          # estatura estimada con el esqueleto (se calcula una vez, al guardar la muestra)


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
        """Estatura más alta reciente: sirve de escala aunque la persona esté agachada o tendida. Se estima con
        el esqueleto, no solo con la caja: si una mesa tapa las piernas, la caja mide medio cuerpo y todas las
        velocidades «en alturas» saldrían el doble de rápidas."""
        v = self.ventana(t_ahora, segundos) or list(self.muestras)
        return float(max(m.alto or alto_estimado(m.caja, m.puntos, m.conf_puntos) for m in v)) if v else 1.0


class Historial:
    def __init__(self, olvido: float = 5.0, unir_hasta: float = 2.0):
        self.personas: dict[int, Pista] = {}
        self.objetos: dict[int, Pista] = {}
        self.clase_objeto: dict[int, str] = {}
        self.olvido = olvido
        self.unir_hasta = unir_hasta          # segundos que se espera a una persona perdida para darle su número
        self.alias: dict[int, int] = {}       # número nuevo del seguidor -> número con el que se la conoce

    def unificar(self, obs: Observacion) -> None:
        """El seguidor le cambia el número a una persona cuando la pierde un instante (se cruza con otra, la tapa
        un mueble, la cámara tiembla o la PC analiza pocos cuadros). Para las reglas sería otra persona: el
        merodeo empezaría a contar de cero, una caída perdería el «antes» y un arrebato, al dueño.

        Si aparece un número nuevo justo donde se perdió hace poco una persona que no está en este cuadro, se le
        devuelve el número anterior (en `obs`, para que reglas, dibujo y alertas usen el mismo)."""
        presentes = set()
        nuevos = []
        for p in obs.personas:
            p.id = self.alias.get(p.id, p.id)
            if p.id in self.personas:
                presentes.add(p.id)
            else:
                nuevos.append(p)
        usados = set()
        for p in nuevos:
            mejor = None
            for pid, pista in self.personas.items():
                hueco = obs.t - pista.ultima_vez
                if pid in presentes or pid in usados or not 0 < hueco <= self.unir_hasta:
                    continue
                m = pista.muestras[-1]
                alto = max(1.0, float(m.caja[3] - m.caja[1]), float(p.caja[3] - p.caja[1]))
                # dónde debería estar ahora, si siguió moviéndose como en el último segundo
                previas = pista.ventana(m.t, 1.0)
                vel = np.zeros(2)
                if len(previas) > 1 and m.t - previas[0].t > 0.05:
                    c0 = (previas[0].caja[:2] + previas[0].caja[2:]) / 2
                    c1 = (m.caja[:2] + m.caja[2:]) / 2
                    vel = (c1 - c0) / (m.t - previas[0].t)
                prevista = m.caja + np.tile(vel * hueco, 2)
                coincide = _iou(p.caja, prevista)
                # o los pies quedaron en el mismo lugar (al caer, la caja cambia de forma pero los pies no se mueven)
                pies_viejos = np.array([(m.caja[0] + m.caja[2]) / 2, m.caja[3]])
                pies = float(np.linalg.norm(p.pies - pies_viejos)) / alto
                if coincide >= 0.5 or (_iou(p.caja, m.caja) >= 0.1 and pies <= 0.25):
                    puntaje = coincide - pies
                    if mejor is None or puntaje > mejor[0]:
                        mejor = (puntaje, pid)
            if mejor is not None:
                self.alias[p.id] = mejor[1]
                p.id = mejor[1]
                usados.add(mejor[1])
        for r in getattr(obs, "rostros", []) or []:
            if r.persona_id is not None:
                r.persona_id = self.alias.get(r.persona_id, r.persona_id)

    def actualizar(self, obs: Observacion) -> None:
        self.unificar(obs)
        for p in obs.personas:
            self._agregar(self.personas, p.id, obs.t, p.caja, p.puntos, p.conf_puntos)
        for o in obs.objetos:
            self._agregar(self.objetos, o.id, obs.t, o.caja, None, None)
            self.clase_objeto[o.id] = o.clase
        for tabla in (self.personas, self.objetos):
            for k in [k for k, v in tabla.items() if obs.t - v.ultima_vez > self.olvido]:
                del tabla[k]
        for k in [k for k, v in self.alias.items() if v not in self.personas]:
            del self.alias[k]

    @staticmethod
    def _agregar(tabla, id_, t, caja, puntos, conf):
        pista = tabla.get(id_)
        if pista is None:
            pista = tabla[id_] = Pista(id=id_, primera_vez=t, ultima_vez=t)
        pista.ultima_vez = t
        caja = np.asarray(caja, float)
        puntos = None if puntos is None else np.asarray(puntos, float)
        conf = None if conf is None else np.asarray(conf, float)
        pista.muestras.append(Muestra(t, caja, puntos, conf, alto_estimado(caja, puntos, conf)))

    def visibles(self, t: float, tabla: str = "personas") -> list[Pista]:
        d = self.personas if tabla == "personas" else self.objetos
        return [p for p in d.values() if t - p.ultima_vez < 0.5]
