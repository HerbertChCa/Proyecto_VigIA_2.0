"""Filtro de objetos detectados: evita que la ropa colgada se lea como «maleta» o una bicicleta como «moto».

El detector de objetos (YOLO entrenado con COCO) duda entre clases parecidas y cambia de opinión de un cuadro
a otro. Este filtro, uno por cámara, decide qué objetos se muestran y se vigilan:

1. Cada objeto seguido acumula votos por clase (la confianza de cada detección). Su nombre es la clase más
   votada, no la del último cuadro: una moto no pasa a «bicicleta» porque un cuadro salga mal.
2. Si duda en serio entre bicicleta y motocicleta, se lo nombra «bicicleta o moto» en vez de adivinar.
3. Un objeto recién aparecido no cuenta hasta confirmarse: varias detecciones seguidas con una confianza
   mínima propia de su clase (más exigente para «maleta» y «bolso», que se confunden con ropa y estantes).
4. Las clases que el operador apaga en Ajustes (por ejemplo, «maleta» en una tienda de ropa) no se usan.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .tipos import Objeto

DOS_RUEDAS = "bicicleta o moto"

# confianza media mínima para confirmar un objeto de cada clase (0 a 1). El detector de vocabulario abierto
# da puntajes más bajos que el de COCO, por eso tiene su propia escala. Una mochila puesta a la espalda o
# sujetada en un tirón sale con poca confianza (0.3-0.4) y casi no se confunde con otra cosa: umbral bajo.
CONFIANZA_MIN_ABIERTO = {"maleta": 0.35, "bolso": 0.30, "ropa": 0.25, "cámara": 0.35, "cuchillo": 0.30}
CONFIANZA_BASE_ABIERTO = 0.25
CONFIANZA_MIN = {"mochila": 0.30, "bolso": 0.50, "maleta": 0.60, "bicicleta": 0.45, "motocicleta": 0.50,
                 DOS_RUEDAS: 0.45, "auto": 0.45, "cuchillo": 0.45, "bate": 0.50}


def en_la_mano(caja, persona) -> bool:
    """¿Una muñeca sujeta la prenda? Quien la lleva en la mano o la muestra la toma por arriba o por los costados
    (muñeca en la mitad superior de la prenda); en una prenda puesta, las manos cuelgan a la altura de su borde
    inferior (la cadera) o por fuera."""
    if persona.puntos is None:
        return False
    x1, y1, x2, y2 = caja
    m = 0.05 * persona.alto
    for k in (9, 10):                                  # muñecas
        q = persona.punto(k)
        if q is not None and x1 - m <= q[0] <= x2 + m and y1 - m <= q[1] <= y1 + 0.6 * (y2 - y1):
            return True
    return False


def _dentro(caja, contenedor) -> float:
    """Fracción del área de caja que cae dentro de contenedor."""
    x1, y1 = max(caja[0], contenedor[0]), max(caja[1], contenedor[1])
    x2, y2 = min(caja[2], contenedor[2]), min(caja[3], contenedor[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1e-6, (caja[2] - caja[0]) * (caja[3] - caja[1]))


class FiltroObjetos:
    """Además de lo anterior, evita que un objeto parpadee (en las pruebas aparecía y desaparecía):
    5. Si el detector le cambia el número a un objeto ya confirmado (pasa cuando lo pierde un instante), el número
       nuevo hereda la confirmación y los votos del anterior, en vez de volver a esperar 3 detecciones.
    6. Si un objeto confirmado no sale en una pasada del detector, se mantiene su última posición hasta `retener`
       segundos: una pasada fallida no lo hace desaparecer.
    """

    def __init__(self, detecciones_min: int = 3, ventana: float = 4.0, abierto: bool = False, retener: float = 1.0):
        self.abierto = abierto
        self.detecciones_min = detecciones_min
        self.ventana = ventana
        self.retener = retener
        self._votos: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._vistas: dict[int, list[tuple[float, float]]] = defaultdict(list)   # (t, confianza)
        self._confirmados: set[int] = set()
        self._ultima: dict[int, float] = {}
        self._ultimo_obj: dict[int, Objeto] = {}       # última versión mostrada de cada objeto confirmado

    def _heredar(self, o: Objeto, t: float, presentes: set[int]) -> bool:
        """¿Es un objeto confirmado que acaba de cambiar de número? Mismo nombre, mismo lugar, visto hace poco."""
        diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
        for oid in list(self._confirmados):
            prev = self._ultimo_obj.get(oid)
            if oid == o.id or oid in presentes or prev is None or not 0 < t - self._ultima.get(oid, -1e9) <= 3.0:
                continue
            if prev.clase not in (o.clase, DOS_RUEDAS) and self.clase(oid) != o.clase:
                continue
            centro = np.array([(prev.caja[0] + prev.caja[2]) / 2, (prev.caja[1] + prev.caja[3]) / 2])
            if float(np.linalg.norm(centro - np.asarray(o.centro))) <= 1.0 * max(diag, 1.0):
                for clase, v in self._votos[oid].items():
                    self._votos[o.id][clase] += v
                self._confirmados.add(o.id)
                for d in (self._votos, self._vistas, self._ultima, self._ultimo_obj):
                    d.pop(oid, None)
                self._confirmados.discard(oid)
                return True
        return False

    def clase(self, oid: int) -> str:
        v = self._votos[oid]
        orden = sorted(v.items(), key=lambda kv: kv[1], reverse=True)
        primera = orden[0][0]
        if len(orden) > 1 and {orden[0][0], orden[1][0]} == {"bicicleta", "motocicleta"} \
                and orden[1][1] >= 0.6 * orden[0][1]:
            return DOS_RUEDAS
        return primera

    def filtrar(self, objetos: list[Objeto], t: float, cfg: dict | None = None, personas=()) -> list[Objeto]:
        cfg = cfg or {}
        activas = cfg.get("clases", {})            # {"maleta": False, ...}; lo que no figura queda activo
        base = CONFIANZA_BASE_ABIERTO if self.abierto else 0.45
        umbrales = {**(CONFIANZA_MIN_ABIERTO if self.abierto else CONFIANZA_MIN), **cfg.get("confianza", {})}
        salida = []
        presentes = {o.id for o in objetos}
        for o in objetos:
            if o.clase == "ropa" and any(_dentro(o.caja, p.caja) >= 0.6 and not en_la_mano(o.caja, p)
                                         for p in personas):
                continue                           # es la ropa que alguien lleva puesta, no un objeto
            nueva = not getattr(o, "_filtrado", False)       # el detector repite su última lista entre detecciones
            o._filtrado = True
            vistas = self._vistas[o.id]
            if nueva:
                self._votos[o.id][o.clase] += o.conf
                vistas.append((t, o.conf))
            self._vistas[o.id] = vistas = [v for v in vistas if t - v[0] <= self.ventana]
            if not vistas:
                continue
            self._ultima[o.id] = t
            if o.id not in self._confirmados and nueva and self._heredar(o, t, presentes):
                pass                                   # mismo objeto, otro número: ya estaba confirmado
            nombre = self.clase(o.id)
            if o.id not in self._confirmados:
                media = sum(c for _, c in vistas) / len(vistas)
                if len(vistas) >= self.detecciones_min and media >= umbrales.get(nombre, base):
                    self._confirmados.add(o.id)
            if o.id not in self._confirmados:
                continue
            apagada = activas.get(nombre) is False or (nombre == DOS_RUEDAS and (
                activas.get("bicicleta") is False and activas.get("motocicleta") is False))
            if apagada:
                continue
            salida.append(Objeto(o.id, nombre, o.caja, o.conf))
            self._ultimo_obj[o.id] = salida[-1]
        # una pasada del detector que no lo vio no lo borra: se mantiene su última posición un momento
        for oid in self._confirmados - presentes:
            prev = self._ultimo_obj.get(oid)
            if prev is not None and 0 < t - self._ultima.get(oid, -1e9) <= self.retener:
                salida.append(Objeto(prev.id, prev.clase, prev.caja, prev.conf))
        for oid in [k for k, u in self._ultima.items() if t - u > 10.0]:      # objetos que ya no se ven
            for d in (self._votos, self._vistas, self._ultima, self._ultimo_obj):
                d.pop(oid, None)
            self._confirmados.discard(oid)
        return salida
