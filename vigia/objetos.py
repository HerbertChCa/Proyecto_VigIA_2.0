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

from .tipos import Objeto

DOS_RUEDAS = "bicicleta o moto"

# confianza media mínima para confirmar un objeto de cada clase (0 a 1)
CONFIANZA_MIN = {"mochila": 0.40, "bolso": 0.50, "maleta": 0.60, "bicicleta": 0.45, "motocicleta": 0.50,
                 DOS_RUEDAS: 0.45, "auto": 0.45, "cuchillo": 0.45, "bate": 0.50}


class FiltroObjetos:
    def __init__(self, detecciones_min: int = 3, ventana: float = 4.0):
        self.detecciones_min = detecciones_min
        self.ventana = ventana
        self._votos: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._vistas: dict[int, list[tuple[float, float]]] = defaultdict(list)   # (t, confianza)
        self._confirmados: set[int] = set()
        self._ultima: dict[int, float] = {}

    def clase(self, oid: int) -> str:
        v = self._votos[oid]
        orden = sorted(v.items(), key=lambda kv: kv[1], reverse=True)
        primera = orden[0][0]
        if len(orden) > 1 and {orden[0][0], orden[1][0]} == {"bicicleta", "motocicleta"} \
                and orden[1][1] >= 0.6 * orden[0][1]:
            return DOS_RUEDAS
        return primera

    def filtrar(self, objetos: list[Objeto], t: float, cfg: dict | None = None) -> list[Objeto]:
        cfg = cfg or {}
        activas = cfg.get("clases", {})            # {"maleta": False, ...}; lo que no figura queda activo
        umbrales = {**CONFIANZA_MIN, **cfg.get("confianza", {})}
        salida = []
        for o in objetos:
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
            nombre = self.clase(o.id)
            if o.id not in self._confirmados:
                media = sum(c for _, c in vistas) / len(vistas)
                if len(vistas) >= self.detecciones_min and media >= umbrales.get(nombre, 0.45):
                    self._confirmados.add(o.id)
            if o.id not in self._confirmados:
                continue
            apagada = activas.get(nombre) is False or (nombre == DOS_RUEDAS and (
                activas.get("bicicleta") is False and activas.get("motocicleta") is False))
            if apagada:
                continue
            salida.append(Objeto(o.id, nombre, o.caja, o.conf))
        for oid in [k for k, u in self._ultima.items() if t - u > 10.0]:      # objetos que ya no se ven
            for d in (self._votos, self._vistas, self._ultima):
                d.pop(oid, None)
            self._confirmados.discard(oid)
        return salida
