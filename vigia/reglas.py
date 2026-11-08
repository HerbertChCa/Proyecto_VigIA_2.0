"""Reglas de comportamiento sobre la pose y el seguimiento.

Cada regla es transparente: trabaja con medidas físicas (velocidad de las extremidades,
ángulo del tronco, tiempo en una zona) y devuelve la evidencia numérica que la disparó.
Así cada alerta se puede explicar, que es uno de los vacíos que señaló la revisión.
"""
from __future__ import annotations

import math
from datetime import datetime, time as hora
from typing import Optional

import numpy as np

from .historial import Historial, Muestra, Pista
from .tipos import (CADERA_D, CADERA_I, CODO_D, CODO_I, EMOCIONES, HOMBRO_D, HOMBRO_I,
                    MUNECA_D, MUNECA_I, NARIZ, OJO_D, OJO_I, OREJA_D, OREJA_I, RODILLA_D, RODILLA_I, TOBILLO_D,
                    TOBILLO_I,
                    Evento, Observacion, alto_estimado)

VEHICULOS = {"bicicleta", "auto", "motocicleta", "bicicleta o moto"}
PORTATILES = {"mochila", "bolso", "maleta", "saco", "bolsa", "caja", "laptop", "celular", "tablet", "cámara",
              "audífonos", "billetera", "ropa"}
# La ropa de una tienda (en percheros o escaparates) no la trajo nadie: queda como escenario y no alerta. Sí se
# vigila una prenda que alguien trajo (una casaca dejada en una silla) o que lleva en la mano.
PRENDAS = {"ropa"}
ARMAS = {"cuchillo", "bate"}

EXTREMIDADES = [MUNECA_I, MUNECA_D, CODO_I, CODO_D, TOBILLO_I, TOBILLO_D]

# arrebato de un objeto que alguien lleva encima (valores por defecto si config.yaml no trae la sección)
ARREBATO = {"activa": True, "tiempo_porte": 1.5, "cambio_min": 0.6, "distancia_mano": 0.45, "separacion": 1.2}
# forcejeo: cuerpo a cuerpo sostenido con movimiento brusco, aunque no haya golpes rápidos
FORCEJEO = {"activa": True, "duracion": 1.5, "brazos": 1.0, "empuje": 0.35}


# --------------------------------------------------------------------------- utilidades
def _medio(m: Muestra, a: int, b: int, conf_min: float = 0.3) -> Optional[np.ndarray]:
    if m.puntos is None:
        return None
    ok = m.conf_puntos is None or (m.conf_puntos[a] >= conf_min and m.conf_puntos[b] >= conf_min)
    return (m.puntos[a] + m.puntos[b]) / 2 if ok else None


def _torso(m: Muestra) -> np.ndarray:
    c = _medio(m, CADERA_I, CADERA_D)
    if c is None:
        c = np.array([(m.caja[0] + m.caja[2]) / 2, (m.caja[1] + m.caja[3]) / 2])
    return c


def velocidades_extremidades(p: Pista, t: float, ventana: float, alto: float) -> list[tuple[float, float]]:
    """(t, velocidad) de la extremidad más rápida relativa al torso, en alturas/s."""
    v = p.ventana(t, ventana)
    out = []
    for m0, m1 in zip(v, v[1:]):
        dt = m1.t - m0.t
        if dt <= 0 or m0.puntos is None or m1.puntos is None:
            continue
        t0, t1 = _torso(m0), _torso(m1)
        mejor = 0.0
        for k in EXTREMIDADES:
            if m0.conf_puntos is not None and (m0.conf_puntos[k] < 0.3 or m1.conf_puntos[k] < 0.3):
                continue
            rel = (m1.puntos[k] - t1) - (m0.puntos[k] - t0)
            mejor = max(mejor, float(np.linalg.norm(rel)) / dt / alto)
        out.append((m1.t, mejor))
    return out


def paso_muestras(muestras) -> float:
    """Segundos típicos entre dos cuadros analizados (0.1 a 10 c/s; 0.7 en una PC que da 1.4 c/s)."""
    ts = [m.t for m in muestras]
    return float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1


def agitacion(p: Pista, t: float, ventana: float = 1.5, puntos=None) -> float:
    """Cuánto se mueven las manos y los codos respecto del torso (desvío de su posición, en alturas). Con pocos
    cuadros por segundo la velocidad no se puede medir (un golpe dura menos que el hueco entre dos cuadros), pero
    un brazo que golpea o forcejea aparece cada vez en un lugar distinto; uno que acompaña la marcha, casi en el mismo."""
    puntos = puntos or (MUNECA_I, MUNECA_D, CODO_I, CODO_D)
    v = [m for m in p.ventana(t, ventana) if m.puntos is not None]
    if len(v) < 3:
        return 0.0
    alto = p.alto_de_pie(t)
    mejor = 0.0
    for k in puntos:
        rel = [m.puntos[k] - _torso(m) for m in v if m.conf_puntos is None or m.conf_puntos[k] >= 0.3]
        if len(rel) >= 3:
            mejor = max(mejor, float(np.std(np.array(rel), axis=0).max()) / alto)
    return mejor


def pies_de(p) -> np.ndarray:
    """Dónde pisa la persona. Si una mesa o el borde de la imagen tapa las piernas, la caja termina en la cintura:
    se estima el piso con la estatura (las zonas se dibujan sobre el piso)."""
    tobillos = [q for q in (p.punto(TOBILLO_I), p.punto(TOBILLO_D)) if q is not None]
    if len(tobillos) == 2:
        return np.array([(p.caja[0] + p.caja[2]) / 2, max(float(np.mean([q[1] for q in tobillos])), p.caja[3])])
    if p.alto - (p.caja[3] - p.caja[1]) > 0.15 * p.alto and p.punto(NARIZ) is not None:
        return np.array([(p.caja[0] + p.caja[2]) / 2, p.caja[1] + p.alto])
    return p.pies


def caja_completa(p) -> np.ndarray:
    """Caja de la persona entera aunque se vea solo de la cintura para arriba."""
    c = np.asarray(p.caja, float).copy()
    if p.alto - (c[3] - c[1]) > 0.15 * p.alto and p.punto(NARIZ) is not None:
        c[3] = c[1] + p.alto
    return c


def zona_cuerpo(persona, margen: float = 0.12):
    """Rectángulo del tronco y la cabeza (nariz, hombros y caderas) ampliado; sin pose, la caja entera."""
    if persona.puntos is not None:
        pts = [persona.punto(i) for i in (0, HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D)]
        pts = [q for q in pts if q is not None]
        if len(pts) >= 3:
            a = np.array(pts)
            m = margen * persona.alto
            return a[:, 0].min() - m, a[:, 1].min() - m, a[:, 0].max() + m, a[:, 1].max() + m
    return tuple(persona.caja)


def contacto(p: Pista, otra, t: float, ventana: float = 1.5) -> bool:
    """¿Una mano o un pie de la persona p llegó al tronco o la cabeza de la otra en la ventana?
    En una pelea hay contacto con el cuerpo; mover los brazos rápido cerca de alguien (por ejemplo,
    sobre una moto) no basta. Se usa el tronco y no la caja, porque la caja incluye los brazos."""
    x1, y1, x2, y2 = zona_cuerpo(otra)
    for m in p.ventana(t, ventana):
        if m.puntos is None:
            continue
        for k in (MUNECA_I, MUNECA_D, TOBILLO_I, TOBILLO_D):
            if m.conf_puntos is not None and m.conf_puntos[k] < 0.3:
                continue
            x, y = m.puntos[k]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
    return False


def fraccion_dentro(caja, contenedor) -> float:
    """Fracción del área de caja que cae dentro de contenedor."""
    ix = max(0.0, min(caja[2], contenedor[2]) - max(caja[0], contenedor[0]))
    iy = max(0.0, min(caja[3], contenedor[3]) - max(caja[1], contenedor[1]))
    return ix * iy / max(1e-6, (caja[2] - caja[0]) * (caja[3] - caja[1]))


def cuerpos_juntos(pa, pb, minimo: float = 0.25) -> bool:
    """¿Los troncos de dos personas se superponen en la imagen? En un forcejeo o un abrazo las manos quedan
    detrás del otro y la pose no las ve: entonces el contacto se nota en que los cuerpos se juntan."""
    a, b = zona_cuerpo(pa, 0.0), zona_cuerpo(pb, 0.0)
    menor = min(fraccion_dentro(a, b), fraccion_dentro(b, a))
    return max(fraccion_dentro(a, b), fraccion_dentro(b, a)) >= minimo and menor > 0


def sostiene(persona, caja, distancia_mano: float = 0.45, clase: str = "") -> bool:
    """¿La persona lleva el objeto encima? En la mano (una muñeca junto al objeto) o pegado al cuerpo
    (una mochila a la espalda o abrazada queda casi entera dentro de la caja de la persona). Para una prenda
    solo cuenta la mano: en una tienda, la ropa colgada queda detrás de los clientes sin que la lleven."""
    if clase not in PRENDAS and fraccion_dentro(caja, persona.caja) >= 0.6:
        return True
    for k in (MUNECA_I, MUNECA_D):
        q = persona.punto(k)
        if q is None:
            continue
        dx = max(caja[0] - q[0], 0.0, q[0] - caja[2])
        dy = max(caja[1] - q[1], 0.0, q[1] - caja[3])
        if math.hypot(dx, dy) / persona.alto <= distancia_mano:
            return True
    return False


def extremidad_en(p: Pista, t: float, dentro, ventana: float = 1.5) -> bool:
    """¿Alguna muñeca o tobillo de p estuvo en la región (función x, y -> bool) en la ventana?"""
    for m in p.ventana(t, ventana):
        if m.puntos is None:
            continue
        for k in (MUNECA_I, MUNECA_D, TOBILLO_I, TOBILLO_D):
            if m.conf_puntos is not None and m.conf_puntos[k] < 0.3:
                continue
            if dentro(*m.puntos[k]):
                return True
    return False


def velocidad_cuerpo(p: Pista, t: float, ventana: float = 1.0) -> float:
    """Desplazamiento del centro de la persona, en alturas por segundo (caminar ≈ 0.6-0.9; correr > 1.6)."""
    v = p.ventana(t, ventana)
    if len(v) < 2 or v[-1].t - v[0].t < 0.5 * ventana:
        return 0.0
    c0 = (v[0].caja[:2] + v[0].caja[2:]) / 2
    c1 = (v[-1].caja[:2] + v[-1].caja[2:]) / 2
    return float(np.linalg.norm(c1 - c0)) / (v[-1].t - v[0].t) / p.alto_de_pie(t)


def zona_activa(z: dict, ahora: hora) -> bool:
    """Una zona sin horario está siempre activa; con horario {desde, hasta} (HH:MM) puede cruzar la medianoche."""
    h = z.get("horario")
    if not h or not h.get("desde") or not h.get("hasta"):
        return True
    d = hora.fromisoformat(h["desde"])
    a = hora.fromisoformat(h["hasta"])
    return d <= ahora < a if d < a else (ahora >= d or ahora < a)


def iou(a, b) -> float:
    """Intersección sobre unión de dos cajas [x1, y1, x2, y2]."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def solape(caja_persona, caja_objeto) -> float:
    """Fracción de la caja de la persona que cae dentro de la caja del objeto."""
    ix = max(0.0, min(caja_persona[2], caja_objeto[2]) - max(caja_persona[0], caja_objeto[0]))
    iy = max(0.0, min(caja_persona[3], caja_objeto[3]) - max(caja_persona[1], caja_objeto[1]))
    area = max(1.0, (caja_persona[2] - caja_persona[0]) * (caja_persona[3] - caja_persona[1]))
    return ix * iy / area


def punto_en_poligono(x: float, y: float, poli: list) -> bool:
    dentro = False
    n = len(poli)
    for i in range(n):
        x1, y1 = poli[i]
        x2, y2 = poli[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                dentro = not dentro
    return dentro


def angulo_tronco(m: Muestra) -> Optional[float]:
    """Ángulo del tronco respecto de la vertical (0 = de pie, 90 = tendido)."""
    h = _medio(m, CADERA_I, CADERA_D)
    s = _medio(m, HOMBRO_I, HOMBRO_D)
    if h is None or s is None:
        return None
    d = s - h                      # de la cadera a los hombros
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return None
    # en la imagen "arriba" es y negativa: 0° = hombros sobre la cadera, 90° = tronco horizontal
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[1] / n))))


def es_horizontal(m: Muestra, umbral: float) -> bool:
    a = angulo_tronco(m)
    if a is not None:
        return a >= umbral
    w, h = m.caja[2] - m.caja[0], m.caja[3] - m.caja[1]
    return w / max(h, 1) > 1.3


def _media(m: Muestra, indices, conf_min: float = 0.3) -> Optional[np.ndarray]:
    if m.puntos is None:
        return None
    pts = [m.puntos[i] for i in indices if m.conf_puntos is None or m.conf_puntos[i] >= conf_min]
    return np.mean(pts, axis=0) if pts else None


CABEZA = (NARIZ, OJO_I, OJO_D, OREJA_I, OREJA_D)


def largo_cuerpo(m: Muestra) -> float:
    """Cabeza-cadera-tobillos medido sobre el cuerpo: casi no cambia al acostarse (sí la altura de la caja)."""
    cab, cad, pies = _media(m, CABEZA), _medio(m, CADERA_I, CADERA_D), _media(m, (TOBILLO_I, TOBILLO_D))
    if cab is None or cad is None or pies is None:
        return 0.0
    return float(np.linalg.norm(cab - cad) + np.linalg.norm(cad - pies))


def esta_tendido(m: Muestra, umbral: float, alto_ref: float) -> bool:
    """¿La persona quedó en el suelo? Se mira cuánto se «aplastó» el cuerpo en vertical respecto de su altura
    de pie: cabeza casi a la altura de la cadera y cadera casi a la altura de los pies. Esa medida no depende
    de hacia dónde mire la cámara ni de hacia dónde cayó la persona (de costado, hacia la cámara o alejándose,
    con la cámara a la altura del pecho o elevada como una de seguridad). Inclinarse a recoger algo no cuenta:
    las piernas siguen sosteniendo la cadera.
    """
    cad = _medio(m, CADERA_I, CADERA_D)
    cab, pies = _media(m, CABEZA), _media(m, (TOBILLO_I, TOBILLO_D))
    alto = max(alto_ref, largo_cuerpo(m) / 0.85, 1.0)
    if cad is not None and pies is not None and (pies[1] - cad[1]) / alto >= 0.35:
        return False                       # de pie, inclinado o agachado: las piernas sostienen la cadera
    if cad is not None and cab is not None and (cad[1] - cab[1]) / alto >= 0.33:
        return False                       # sentado o en cuclillas: el tronco sigue erguido sobre la cadera
    rodillas = _media(m, (RODILLA_I, RODILLA_D))
    if rodillas is not None and pies is not None and (pies[1] - rodillas[1]) / alto >= 0.18:
        return False                       # agachado para recoger algo: las canillas siguen verticales
    if cad is not None and cab is not None and pies is not None:
        return True                        # cuerpo aplastado en vertical: tendido, mire de donde mire la cámara
    return es_horizontal(m, umbral)        # faltan puntos (persona tapada o lejos): ángulo del tronco o caja


# --------------------------------------------------------------------------- reglas
class Reglas:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._puestos: list[dict] = []   # vehículos estacionados vistos por esta cámara
        self._contactos: dict = {}       # (quien toca, a quien) -> (instante, distancia) para la regla de huida
        self._ultimo_toque: dict = {}    # (quien toca, a quien) -> último instante con mano o pie en el cuerpo
        self._juntos: dict = {}          # (a, b) con a < b -> [(t, ¿cuerpo a cuerpo?)] de los últimos segundos
        self._portes: list[dict] = []    # objetos que alguien lleva encima, para detectar un arrebato
        self._n_porte = 0
        self._multitud_desde = None
        self.reloj = lambda: datetime.now().time()   # se reemplaza en las pruebas
        self._t0 = None                  # primer cuadro analizado: lo que ya estaba es parte del escenario
        self._escenario: list = []       # cajas de objetos que pertenecen al lugar (escaparates, estantes…)
        self.ignorar: list = []          # cajas que el operador marcó como falsa alarma

    # ---- escenario aprendido
    def aprender_ignorar(self, caja) -> None:
        """El operador marcó una alerta como falsa: ese objeto o vehículo no vuelve a alertar en ese lugar."""
        caja = [float(v) for v in caja]
        if not any(iou(caja, c) >= 0.6 for c in self.ignorar):
            self.ignorar.append(caja)

    def _ignorado(self, caja) -> bool:
        return any(iou(caja, c) >= 0.4 for c in self.ignorar)

    def _es_escenario(self, caja) -> bool:
        return any(iou(caja, c) >= 0.4 for c in self._escenario)

    def _recordar_escenario(self, caja) -> None:
        caja = [float(v) for v in caja]
        if not self._es_escenario(caja) and len(self._escenario) < 300:
            self._escenario.append(caja)

    def actualizar_cfg(self, cfg: dict):
        self.cfg = cfg

    def duenos(self, hist: Historial) -> dict[int, int]:
        """Número de seguimiento de cada objeto -> número de la persona dueña (quien lo trajo o lo lleva encima).
        Se muestra en el video («mochila #12 de #3») para ver a quién asoció VigIA cada objeto."""
        out = {oid: p.estado["dueno"] for oid, p in hist.objetos.items() if p.estado.get("dueno") is not None}
        for r in self._portes:
            if r["dueno"] is not None:
                for oid in r["oids"]:
                    out.setdefault(oid, r["dueno"])
        return out

    def evaluar(self, obs: Observacion, hist: Historial, ancho: int, alto: int,
                zonas: list[dict]) -> list[Evento]:
        ev: list[Evento] = []
        r = self.cfg
        if self._t0 is None:
            self._t0 = obs.t
        self._hist = hist                      # para la segunda mirada (vigia/revision.py: sospecha)
        self._rap = self._rapidez(obs, hist)
        self._registrar_contactos(obs, hist)
        if r["violencia"]["activa"]:
            ev += self._violencia(obs, hist)
        if r.get("huida", {}).get("activa"):
            ev += self._huida(obs, hist)
        if r.get("vandalismo", {}).get("activa"):
            ev += self._vandalismo(obs, hist, ancho, alto, zonas)
        if r.get("armas", {}).get("activa"):
            ev += self._armas(obs, hist)
        if r.get("aglomeracion", {}).get("activa"):
            ev += self._aglomeracion(obs, hist)
        if r["caida"]["activa"]:
            ev += self._caida(obs, hist)
        if r["merodeo"]["activa"] or r["intrusion"]["activa"]:
            ev += self._zonas(obs, hist, ancho, alto, zonas)
        if r["objetos"]["activa"]:
            ev += self._objetos(obs, hist)
            if {**ARREBATO, **r.get("arrebato", {})}["activa"]:
                ev += self._arrebato(obs, hist)
        if r.get("vehiculos", {}).get("activa"):
            ev += self._vehiculos(obs)
        ev += self._emociones(obs, hist)
        return ev

    # ---- rapidez de las extremidades, compartida por violencia y vandalismo
    def _rapidez(self, obs, hist) -> dict:
        """pid -> (segundos con extremidades rápidas en los últimos 1.5 s, velocidad máxima)."""
        umbral = self.cfg["violencia"]["velocidad_extremidades"]
        out = {}
        for p in obs.personas:
            pista = hist.personas.get(p.id)
            if pista is None:
                continue
            vel = velocidades_extremidades(pista, obs.t, 1.5, pista.alto_de_pie(obs.t))
            # cada muestra rápida aporta el intervalo típico entre frames
            paso = float(np.median(np.diff([ti for ti, _ in vel]))) if len(vel) > 1 else 0.125
            dur = paso * sum(1 for _, v in vel if v >= umbral)
            pista.estado["vel_max"] = max((v for _, v in vel), default=0.0)
            pista.estado.pop("agitacion", None)
            if paso >= 0.2:
                # PC lenta (menos de 5 c/s): un golpe cae entre dos cuadros y su velocidad no se ve. Se mide en
                # cambio cuánto cambian de lugar las manos y codos respecto del torso en 1.5 s.
                ag = agitacion(pista, obs.t, max(1.5, 3.5 * paso))
                if ag >= self.cfg["violencia"].get("agitacion_lenta", 0.11):
                    pista.estado["agitacion"] = ag
                    dur = max(dur, self.cfg["violencia"]["duracion_min"])
            out[p.id] = (dur, pista.estado["vel_max"])
        return out

    # ---- contactos entre personas, compartidos por violencia, forcejeo, derribo y huida
    def _registrar_contactos(self, obs, hist) -> None:
        t = obs.t
        act = {p.id: p for p in obs.personas if p.id in hist.personas}
        cerca = []
        for a, pa in act.items():
            for b, pb in act.items():
                if a == b:
                    continue
                dist = float(np.linalg.norm(pa.centro - pb.centro)) / ((pa.alto + pb.alto) / 2)
                if dist > 1.2:
                    continue
                if a < b:
                    cerca.append((a, b))
                if contacto(hist.personas[a], pb, t, ventana=0.5):
                    self._contactos[(a, b)] = (t, dist)
                    self._ultimo_toque[(a, b)] = t
        for a, b in cerca:
            cuerpo = cuerpos_juntos(act[a], act[b])
            if cuerpo:
                self._ultimo_toque[(a, b)] = t                  # cuerpo a cuerpo también cuenta para el derribo
            toque = cuerpo or self._ultimo_toque.get((b, a)) == t
            self._juntos.setdefault((a, b), []).append((t, toque))
        for k in [k for k, (tc, _) in self._contactos.items() if t - tc > 10]:
            del self._contactos[k]
        for k in [k for k, tc in self._ultimo_toque.items() if t - tc > 10]:
            del self._ultimo_toque[k]
        for k in list(self._juntos):
            self._juntos[k] = [m for m in self._juntos[k] if t - m[0] <= 4.0]
            if not self._juntos[k]:
                del self._juntos[k]

    def _forcejeo(self, a, b, pistas, t) -> Optional[str]:
        """Cuerpo a cuerpo sostenido con movimiento brusco (empujones, tirones, agarres) aunque no haya golpes
        rápidos. Un abrazo no cuenta: los cuerpos están juntos pero quietos. Devuelve la evidencia o None."""
        f = {**FORCEJEO, **self.cfg["violencia"].get("forcejeo", {})}
        if not f["activa"]:
            return None
        paso = paso_muestras(pistas[a].ventana(t, 3.0))
        duracion = max(f["duracion"], 3.5 * paso)            # con pocos cuadros por segundo, una ventana más larga
        muestras = [m for m in self._juntos.get((min(a, b), max(a, b)), []) if t - m[0] <= duracion]
        if len(muestras) < 3 or t - muestras[0][0] < 0.8 * duracion - 0.5 * paso:
            return None
        juntos = sum(1 for _, j in muestras if j) / len(muestras)
        if juntos < 0.7:
            return None
        # posición de una persona respecto de la otra, cuadro a cuadro: en un forcejeo va y viene (empujones,
        # tirones) pero no se aleja; dos personas que se cruzan caminando también se tapan, pero pasan de largo
        pa = {round(m.t, 3): _torso(m) for m in pistas[a].ventana(t, duracion)}
        pb = {round(m.t, 3): _torso(m) for m in pistas[b].ventana(t, duracion)}
        comunes = sorted(set(pa) & set(pb))
        if len(comunes) < 3:
            return None
        alto = (pistas[a].alto_de_pie(t) + pistas[b].alto_de_pie(t)) / 2
        rel = [(pa[k] - pb[k]) / alto for k in comunes]
        recorrido = sum(float(np.linalg.norm(q - p)) for p, q in zip(rel, rel[1:]))
        neto = float(np.linalg.norm(rel[-1] - rel[0]))
        if neto > 0.5 or neto > 0.6 * recorrido:
            return None                                        # pasaron de largo o se separaron
        # caminando juntos (de la mano, abrazados, uno al lado del otro) los cuerpos se tapan y no se separan, y los
        # brazos se balancean rápido; pero los dos avanzan a la par. En un forcejeo se quedan casi en el mismo lugar.
        comun = (self._vel_vector(pistas[a], t, duracion) + self._vel_vector(pistas[b], t, duracion)) / 2
        if float(np.linalg.norm(comun)) >= 0.35:
            return None
        empuje = recorrido / max(comunes[-1] - comunes[0], 1e-6)
        brazos = max(self._rapidez_brazos(pistas[a], t), self._rapidez_brazos(pistas[b], t))
        if paso >= 0.2:
            # PC lenta: el vaivén de un empujón dura menos que el hueco entre cuadros y la velocidad sale baja.
            # Se mide cuánto cambia de lugar un cuerpo respecto del otro y los brazos respecto de su torso.
            vaiven = float(np.std(np.array(rel), axis=0).max())
            ag = max(agitacion(pistas[a], t, duracion), agitacion(pistas[b], t, duracion))
            if vaiven < f.get("vaiven_lento", 0.04) and ag < self.cfg["violencia"].get("agitacion_lenta", 0.11):
                return None
            return (f"Cuerpo a cuerpo durante {t - muestras[0][0]:.1f} s ({juntos:.0%} del tiempo), sin separarse, con "
                    f"un vaivén de {vaiven:.2f} alturas entre ambos cuerpos y brazos que cambian de lugar {ag:.2f} "
                    f"alturas (umbrales {f.get('vaiven_lento', 0.04)} y "
                    f"{self.cfg['violencia'].get('agitacion_lenta', 0.11)}; medido con {1 / paso:.1f} cuadros por "
                    "segundo, sin velocidades).")
        if brazos < f["brazos"] and empuje < f["empuje"]:
            return None                                        # juntos y quietos: un abrazo, una conversación
        return (f"Cuerpo a cuerpo durante {t - muestras[0][0]:.1f} s ({juntos:.0%} del tiempo) con movimiento brusco: "
                f"brazos a {brazos:.1f} alturas/s y empujones de {empuje:.2f} alturas/s entre ambos cuerpos "
                f"(umbrales {f['brazos']} y {f['empuje']}), sin separarse.")

    @staticmethod
    def _rapidez_brazos(p: Pista, t: float) -> float:
        """Velocidad máxima de muñecas y codos respecto del torso en el último segundo (sin los pies: al
        caminar, los tobillos ya se mueven rápido respecto de la cadera)."""
        v = p.ventana(t, 1.0)
        mejor = 0.0
        alto = p.alto_de_pie(t)
        for m0, m1 in zip(v, v[1:]):
            dt = m1.t - m0.t
            if dt <= 0 or m0.puntos is None or m1.puntos is None:
                continue
            t0, t1 = _torso(m0), _torso(m1)
            for k in (MUNECA_I, MUNECA_D, CODO_I, CODO_D):
                if m0.conf_puntos is not None and (m0.conf_puntos[k] < 0.3 or m1.conf_puntos[k] < 0.3):
                    continue
                rel = (m1.puntos[k] - t1) - (m0.puntos[k] - t0)
                mejor = max(mejor, float(np.linalg.norm(rel)) / dt / alto)
        return mejor

    @staticmethod
    def _vel_vector(p: Pista, t: float, ventana: float = 1.0) -> np.ndarray:
        v = p.ventana(t, ventana)
        if len(v) < 2 or v[-1].t - v[0].t < 0.3:
            return np.zeros(2)
        return (_torso(v[-1]) - _torso(v[0])) / (v[-1].t - v[0].t) / p.alto_de_pie(t)

    # ---- violencia: dos personas cerca con extremidades muy rápidas y contacto, o en forcejeo
    def _violencia(self, obs, hist) -> list[Evento]:
        c = self.cfg["violencia"]
        t = obs.t
        pistas = {p.id: hist.personas[p.id] for p in obs.personas if p.id in hist.personas}
        actuales = {p.id: p for p in obs.personas if p.id in pistas}
        rapidos = self._rap
        eventos = []
        ids = sorted(actuales)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = actuales[a], actuales[b]
                escala = (pa.alto + pb.alto) / 2
                dist = float(np.linalg.norm(pa.centro - pb.centro)) / escala
                if dist > c["distancia_max"]:
                    continue
                dur = max(rapidos.get(a, (0, 0))[0], rapidos.get(b, (0, 0))[0])
                vmax = max(rapidos.get(a, (0, 0))[1], rapidos.get(b, (0, 0))[1])
                golpes = dur >= c["duracion_min"] and (not c.get("requiere_contacto", True) or
                                                      contacto(pistas[a], pb, t) or contacto(pistas[b], pa, t))
                if golpes:
                    ag = max(pistas[a].estado.get("agitacion", 0.0), pistas[b].estado.get("agitacion", 0.0))
                    medida = (f"Brazos que cambian de lugar {ag:.2f} alturas respecto del torso (umbral "
                              f"{c.get('agitacion_lenta', 0.11)}; con pocos cuadros por segundo no se puede medir la "
                              "velocidad de cada golpe)" if ag and vmax < c["velocidad_extremidades"] else
                              f"Extremidades a {vmax:.1f} alturas/s (umbral {c['velocidad_extremidades']}) durante "
                              f"{dur:.1f} s")
                    eventos.append(Evento(
                        "violencia", "alta", f"Posible pelea entre las personas #{a} y #{b}",
                        f"{medida}, con {dist:.2f} alturas de separación y contacto de una mano o un pie "
                        "con el tronco o la cabeza de la otra persona.",
                        f"violencia:{a}-{b}", [pa.caja.tolist(), pb.caja.tolist()]))
                    continue
                forcejeo = self._forcejeo(a, b, pistas, t)
                if forcejeo:
                    eventos.append(Evento(
                        "violencia", "alta", f"Posible pelea o forcejeo entre las personas #{a} y #{b}",
                        forcejeo, f"violencia:{a}-{b}", [pa.caja.tolist(), pb.caja.tolist()]))
        return eventos

    def _quien_derribo(self, victima: int, desde: float, hasta: float) -> Optional[int]:
        """Persona que tuvo contacto con el cuerpo de la víctima (o cuerpo a cuerpo) justo antes de que cayera."""
        mejor = None
        for (a, b), tc in self._ultimo_toque.items():
            otro = a if b == victima else b if a == victima else None
            if otro is not None and desde <= tc <= hasta and (mejor is None or tc > mejor[0]):
                mejor = (tc, otro)
        return None if mejor is None else mejor[1]

    # ---- caídas y persona en el suelo
    def _caida(self, obs, hist) -> list[Evento]:
        c = self.cfg["caida"]
        t = obs.t
        out = []
        for p in obs.personas:
            pista = hist.personas.get(p.id)
            if pista is None or not pista.muestras:
                continue
            est = pista.estado.setdefault("caida", {"horiz_desde": None, "vert_desde": t,
                                                     "caida": False, "suelo": False})
            ultima = pista.muestras[-1]
            # la altura de pie se fija al empezar a caer: tendida en el suelo, la caja ya no sirve de escala
            alto = est["alto_ref"] if est["horiz_desde"] is not None and est.get("alto_ref") else pista.alto_de_pie(t, 6.0)
            if esta_tendido(ultima, c["angulo_horizontal"], alto):
                if est["horiz_desde"] is None:
                    est["horiz_desde"], est["alto_ref"] = t, alto
                est["vert_desde"] = None
            else:
                if est["vert_desde"] is None:
                    est["vert_desde"] = t
                if t - est["vert_desde"] >= 1.0:   # se levantó: se reinicia el episodio
                    est.update(horiz_desde=None, caida=False, suelo=False)
                continue
            tendido = t - est["horiz_desde"]
            alto = max(est["alto_ref"], largo_cuerpo(ultima) / 0.85, 1.0)
            if not est["caida"] and tendido >= 1.0:    # agacharse un instante no cuenta: en el suelo 1 s

                previas = [m for m in pista.muestras if est["horiz_desde"] - 2.5 <= m.t <= est["horiz_desde"]]
                if previas:
                    ys = [_torso(m)[1] for m in previas]
                    descenso = (_torso(ultima)[1] - min(ys)) / alto
                    if descenso >= c["descenso_min"]:
                        est["caida"] = True
                        out.append(Evento(
                            "caida", "alta", f"Caída de la persona #{p.id}",
                            f"El centro de la cadera bajó {descenso:.2f} alturas en menos de 2.5 s y el "
                            f"cuerpo quedó a ras del suelo (cabeza y cadera casi a la altura de los pies).",
                            f"caida:{p.id}", [p.caja.tolist()]))
                        # ¿alguien la tenía agarrada o la tocó justo antes? Entonces no fue un accidente
                        agresor = self._quien_derribo(p.id, est["horiz_desde"] - 3.0, t)
                        if agresor is not None and self.cfg["violencia"]["activa"]:
                            otro = next((q for q in obs.personas if q.id == agresor), None)
                            out.append(Evento(
                                "violencia", "alta", f"Posible agresión: la persona #{agresor} derribó a #{p.id}",
                                f"La persona #{agresor} estuvo en contacto con el cuerpo de #{p.id} (mano, pie o cuerpo "
                                f"a cuerpo) en los 3 s previos a la caída, y #{p.id} quedó en el suelo.",
                                f"violencia:derribo:{agresor}:{p.id}",
                                [p.caja.tolist()] + ([otro.caja.tolist()] if otro is not None else [])))
            if not est["suelo"] and tendido >= c["en_suelo_alerta"]:
                est["suelo"] = True
                out.append(Evento(
                    "en_suelo", "alta", f"Persona #{p.id} tendida en el suelo",
                    f"Lleva {tendido:.0f} s en posición horizontal sin incorporarse.",
                    f"suelo:{p.id}", [p.caja.tolist()]))
        return out

    # ---- merodeo e intrusión
    def _zonas(self, obs, hist, ancho, alto, zonas) -> list[Evento]:
        c = self.cfg
        t = obs.t
        out = []
        zonas_merodeo = [z for z in zonas if z.get("tipo") == "merodeo"]
        ahora = self.reloj()
        zonas = [z for z in zonas if zona_activa(z, ahora)]
        for p in obs.personas:
            pista = hist.personas.get(p.id)
            if pista is None:
                continue
            pies = pies_de(p)
            fx, fy = pies[0] / ancho, pies[1] / alto
            desde = pista.estado.setdefault("zona_desde", {})
            fuera = pista.estado.setdefault("zona_fuera", {})
            tolerancia = c["merodeo"].get("tolerancia_salida", 2.0)
            for z in zonas:
                dentro = punto_en_poligono(fx, fy, z["puntos"])
                if not dentro:
                    # salir un instante (la cámara tiembla, los pies pisan el borde) no reinicia el merodeo
                    if z["id"] in desde and t - fuera.setdefault(z["id"], t) < tolerancia:
                        continue
                    desde.pop(z["id"], None)
                    fuera.pop(z["id"], None)
                    continue
                fuera.pop(z["id"], None)
                desde.setdefault(z["id"], t)
                if z.get("tipo") == "restringida" and c["intrusion"]["activa"]:
                    out.append(Evento(
                        "intrusion", "alta", f"Intrusión en «{z['nombre']}» (persona #{p.id})",
                        "Los pies de la persona están dentro del polígono de la zona restringida"
                        + (f" durante su horario ({z['horario']['desde']}-{z['horario']['hasta']})." if z.get("horario") else "."),
                        f"intrusion:{z['id']}:{p.id}", [p.caja.tolist()]))
                elif z.get("tipo") == "merodeo" and c["merodeo"]["activa"]:
                    estancia = t - desde[z["id"]]
                    if estancia >= c["merodeo"]["tiempo_max"]:
                        out.append(Evento(
                            "merodeo", "media", f"Merodeo en «{z['nombre']}» (persona #{p.id})",
                            f"Permanece {estancia:.0f} s en la zona (límite {c['merodeo']['tiempo_max']} s).",
                            f"merodeo:{z['id']}:{p.id}", [p.caja.tolist()]))
            if not zonas_merodeo and c["merodeo"]["activa"] and c["merodeo"].get("sin_zonas", False):
                estancia = t - pista.primera_vez
                if estancia >= c["merodeo"]["tiempo_max"]:
                    out.append(Evento(
                        "merodeo", "media", f"Merodeo: persona #{p.id} permanece en la escena",
                        f"Visible durante {estancia:.0f} s (límite {c['merodeo']['tiempo_max']} s; "
                        "sin zonas de merodeo definidas, se mide en toda la cámara).",
                        f"merodeo:cam:{p.id}", [p.caja.tolist()]))
        return out

    # ---- objetos abandonados y posible hurto
    @staticmethod
    def _quien_paso_cerca(hist, centro, desde: float, hasta: float, limite: float = 1.0):
        """Persona que estuvo más cerca de un punto entre dos instantes (en alturas de esa persona).
        Sirve para saber quién dejó un objeto aunque su cuerpo lo tapara mientras lo dejaba."""
        mejor = None
        for pista in hist.personas.values():
            for m in pista.muestras:
                if desde <= m.t <= hasta:
                    alto = alto_estimado(m.caja, m.puntos, m.conf_puntos)
                    cx, cy = (m.caja[0] + m.caja[2]) / 2, (m.caja[1] + m.caja[3]) / 2
                    d = float(np.hypot(cx - centro[0], cy - centro[1])) / alto
                    if d <= limite and (mejor is None or d < mejor[0]):
                        mejor = (d, pista.id)
        return None if mejor is None else mejor[1]

    def _perdidos(self, obs, hist) -> list[Evento]:
        """Objetos quietos y con dueño que dejaron de verse. Al levantarlo, la mano y el cuerpo tapan el objeto
        y el detector lo pierde: si quien estaba al lado no es el dueño y se alejó, se lo llevó."""
        t, out = obs.t, []
        visibles = [o for o in obs.objetos if o.clase in PORTATILES]
        for oid, pista in hist.objetos.items():
            est = pista.estado
            perdido = t - pista.ultima_vez
            if (est.get("resuelto") or not est.get("estuvo_quieto") or perdido < 0.8 or perdido > 4.5
                    or hist.clase_objeto.get(oid) not in PORTATILES):
                continue
            ultima = pista.muestras[-1]
            centro = np.array([(ultima.caja[0] + ultima.caja[2]) / 2, (ultima.caja[1] + ultima.caja[3]) / 2])
            if any(iou(o.caja, ultima.caja) >= 0.2 for o in visibles):
                est["resuelto"] = True              # el detector lo volvió a ver con otro número: sigue ahí
                continue
            cerca = est.get("ultimo_cerca")         # (persona, distancia) cuando se vio por última vez
            if not cerca or cerca[1] > 1.2:
                continue                            # nadie al lado: se tapó o el detector falló; se espera
            quien = hist.personas.get(cerca[0])
            if quien is not None and t - quien.ultima_vez < 0.5:
                m = quien.muestras[-1]
                alto = alto_estimado(m.caja, m.puntos, m.conf_puntos)
                lejos = np.hypot((m.caja[0] + m.caja[2]) / 2 - centro[0], (m.caja[1] + m.caja[3]) / 2 - centro[1]) / alto
                if lejos < 1.0:
                    continue                        # sigue parado ahí (puede estar tapándolo): aún no se decide
            est["resuelto"] = True
            clase, dueno = hist.clase_objeto.get(oid, "objeto"), est.get("dueno")
            if dueno is not None and cerca[0] != dueno:
                out.append(Evento(
                    "hurto", "alta", f"Posible hurto: la persona #{cerca[0]} se lleva una {clase}",
                    f"La {clase} estaba inmóvil y asociada a la persona #{dueno}; dejó de verse cuando la persona "
                    f"#{cerca[0]} estaba a {cerca[1]:.1f} alturas, y esa persona se alejó.",
                    f"hurto:{oid}:{cerca[0]}", [ultima.caja.tolist()]))
            elif est.get("abandonado"):
                out.append(Evento(
                    "hurto", "baja", f"La persona #{cerca[0]} retiró una {clase} abandonada",
                    "El objeto había sido marcado como abandonado y dejó de verse junto a esa persona.",
                    f"retiro:{oid}:{cerca[0]}", [ultima.caja.tolist()]))
        return out

    def _heredar(self, o, est, hist, t) -> bool:
        """Un objeto que aparece de golpe junto a donde se perdió otro quieto es el mismo objeto con otro número
        de seguimiento (pasa al levantarlo): conserva su dueño y su historia."""
        diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
        for oid, pista in hist.objetos.items():
            pe = pista.estado
            if (oid == o.id or pe.get("resuelto") or not pe.get("estuvo_quieto") or "dueno" not in pe
                    or not 0.1 < t - pista.ultima_vez <= 3.0 or hist.clase_objeto.get(oid) not in PORTATILES):
                continue
            u = pista.muestras[-1].caja
            if np.hypot((u[0] + u[2]) / 2 - o.centro[0], (u[1] + u[3]) / 2 - o.centro[1]) <= 2.0 * diag:
                pe["resuelto"] = True
                est.update(origen="nuevo", visto_desde=pe.get("visto_desde", t), dueno=pe["dueno"],
                           estuvo_quieto=True, abandonado=pe.get("abandonado", False), heredado_de=oid)
                return True
        return False

    @staticmethod
    def _encima_de(per, o) -> bool:
        """¿La persona lleva el objeto puesto o agarrado (no apoyado en el piso a sus pies)?"""
        pies = pies_de(per)
        if fraccion_dentro(o.caja, per.caja) >= 0.6 and o.caja[3] < pies[1] - 0.2 * per.alto:
            return True                                   # a la espalda, colgado del hombro, abrazado
        for k in (MUNECA_I, MUNECA_D):
            q = per.punto(k)
            if q is not None and o.caja[0] - 0.05 * per.alto <= q[0] <= o.caja[2] + 0.05 * per.alto \
                    and o.caja[1] - 0.05 * per.alto <= q[1] <= o.caja[3] + 0.05 * per.alto:
                return True                               # la mano dentro del objeto: lo tiene agarrado
        return False

    def _objetos(self, obs, hist) -> list[Evento]:
        c = self.cfg["objetos"]
        t = obs.t
        out = self._perdidos(obs, hist)
        for o in obs.objetos:
            pista = hist.objetos.get(o.id)
            if pista is None or o.clase not in PORTATILES:   # vehículos y armas tienen sus propias reglas
                continue
            est = pista.estado
            if self._ignorado(o.caja):
                continue
            # ¿llegó con alguien o ya era parte del lugar? Solo un objeto que alguien trajo puede quedar abandonado:
            # lo que se ve desde el inicio, lo que ya se recordaba en ese sitio o lo que aparece sin nadie al lado
            # (un estante, una prenda en un escaparate) es escenario.
            if "origen" not in est and not self._heredar(o, est, hist, t):
                # alguien lo lleva encima (en la mano o a la espalda): lo trajo esa persona, aunque VigIA recién
                # arranque o el objeto aparezca donde antes hubo otro. Así una prueba que empieza con la mochila
                # puesta, o que se repite en el mismo lugar, no se toma como parte del escenario.
                encima = any(sostiene(per, o.caja, clase=o.clase) for per in obs.personas)
                ya_estaba = not encima and (t - self._t0 < c.get("calentamiento", 10) or self._es_escenario(o.caja)
                                            or o.clase in PRENDAS)   # una prenda la trae alguien en la mano;
                                                                     # si aparece suelta, es la de la tienda
                est["origen"], est["visto_desde"] = ("escenario" if ya_estaba else "nuevo"), t
            diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
            if est["origen"] == "nuevo" and o.clase in PRENDAS and not est.get("traida"):
                # una prenda que alguien trajo llega en movimiento; la de un perchero o un escaparate solo la tocan
                # los clientes al mirarla y no se mueve: a los 5 s sin moverse pasa a ser parte del lugar
                ini = est.setdefault("centro_inicial", o.centro.copy())
                if float(np.linalg.norm(o.centro - ini)) >= 0.5 * diag:
                    est["traida"] = True
                elif t - est["visto_desde"] > 5.0:
                    est["origen"] = "escenario"
            if est["origen"] == "escenario":
                self._recordar_escenario(o.caja)
                continue
            ultimos = pista.ventana(t, max(1.5, 2.5 * paso_muestras(pista.muestras)))
            # se compara la posición media de la primera y la segunda mitad de la ventana: el temblor de una cámara
            # en la mano mueve la caja unos píxeles en cada cuadro, pero en promedio queda en el mismo lugar
            if len(ultimos) > 1:
                mitad = len(ultimos) // 2
                c0 = np.mean([(m.caja[:2] + m.caja[2:]) / 2 for m in ultimos[:mitad]], axis=0)
                c1 = np.mean([(m.caja[:2] + m.caja[2:]) / 2 for m in ultimos[mitad:]], axis=0)
                mov = float(np.linalg.norm(c1 - c0))
            else:
                mov = 0.0
            quieto = mov < 0.35 * diag
            if quieto:
                est.pop("mov_desde", None)
            elif mov < 1.0 * diag and t - est.setdefault("mov_desde", t) < 1.0:
                quieto = True            # un sacudón breve (cámara en la mano) no es que alguien lo movió

            def cercania(per):
                return float(np.linalg.norm(per.centro - o.centro)) / per.alto

            cerca = sorted(((cercania(per), per) for per in obs.personas), key=lambda x: x[0])
            est["ultimo_cerca"] = (cerca[0][1].id, cerca[0][0]) if cerca else None
            if "dueno" not in est:
                # quien estaba al lado cuando apareció lo trajo; también vale quien pasó junto a ese lugar justo
                # antes, porque al dejarlo su propio cuerpo suele tapar el objeto
                quien = (cerca[0][1].id if cerca and cerca[0][0] <= 1.0 else
                         self._quien_paso_cerca(hist, o.centro, est["visto_desde"] - 3.0, t))
                if quien is not None and t - est["visto_desde"] <= 3.0:
                    est["dueno"] = quien
                elif t - est["visto_desde"] > 3.0:
                    est["origen"] = "escenario"            # apareció sin nadie cerca: no lo trajo nadie
                    self._recordar_escenario(o.caja)
                    continue
            # una mochila a la espalda o en la mano de alguien que está parado también está «quieta», pero no la
            # dejaron: no cuenta para abandono ni para el hurto «del piso» (de eso se encarga la regla de arrebato)
            if quieto and any(self._encima_de(per, o) for per in obs.personas):
                est.pop("quieto_desde", None)
                est.pop("solo_desde", None)
                continue
            if quieto:
                # estuvo quieto toda la ventana que se acaba de medir, no recién ahora
                est.setdefault("quieto_desde", ultimos[0].t if len(ultimos) > 1 else t)
                tiempo_quieto = t - est["quieto_desde"]
                if tiempo_quieto >= 3:
                    est["estuvo_quieto"] = True
                nadie = not cerca or cerca[0][0] > c["distancia_dueno"]
                if nadie:
                    est.setdefault("solo_desde", t)
                else:
                    est.pop("solo_desde", None)
                solo = t - est.get("solo_desde", t)
                if tiempo_quieto >= c["tiempo_abandono"] and solo >= min(5.0, c["tiempo_abandono"] / 3):
                    est["abandonado"] = True
                    dueno = f" (dejado por la persona #{est['dueno']})" if "dueno" in est else ""
                    out.append(Evento(
                        "abandono", "media", f"{o.clase.capitalize()} abandonada{dueno}",
                        f"Inmóvil {tiempo_quieto:.0f} s y sin nadie a menos de {c['distancia_dueno']} "
                        f"alturas durante {solo:.0f} s.",
                        f"abandono:{o.id}", [o.caja.tolist()]))
            else:
                est.pop("quieto_desde", None)
                if est.get("estuvo_quieto") and cerca and cerca[0][0] <= 1.2:
                    portador = cerca[0][1].id
                    dueno = est.get("dueno")
                    if dueno is not None and portador != dueno:
                        out.append(Evento(
                            "hurto", "alta", f"Posible hurto: la persona #{portador} se lleva una {o.clase}",
                            f"El objeto estaba inmóvil y asociado a la persona #{dueno}; ahora se desplaza "
                            f"junto a la persona #{portador}.",
                            f"hurto:{o.id}:{portador}", [o.caja.tolist(), cerca[0][1].caja.tolist()]))
                    elif est.get("abandonado"):
                        out.append(Evento(
                            "hurto", "baja", f"La persona #{portador} retiró una {o.clase} abandonada",
                            "El objeto había sido marcado como abandonado y empezó a moverse.",
                            f"retiro:{o.id}:{portador}", [o.caja.tolist()]))
                    est["estuvo_quieto"] = False
        return out

    # ---- arrebato: alguien le quita a otra persona un objeto que esta llevaba encima
    def _arrebato(self, obs, hist) -> list[Evento]:
        """La regla de hurto vigila objetos que quedaron quietos (en el piso, sobre una mesa). En un arrebato el
        objeto nunca está quieto: va en la mano o a la espalda de su dueño y pasa a las manos de otra persona,
        a veces con forcejeo, y esa persona se aleja. Se sigue a cada objeto por su lugar y no solo por su número
        de seguimiento, porque en el tirón el detector lo pierde o le cambia el número."""
        c = {**ARREBATO, **self.cfg.get("arrebato", {})}
        t, out = obs.t, []
        personas = {p.id: p for p in obs.personas}
        vistos = set()
        for o in obs.objetos:
            if o.clase not in PORTATILES or self._ignorado(o.caja):
                continue
            diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
            quienes = tuple(sorted(pid for pid, p in personas.items()
                                   if sostiene(p, o.caja, c["distancia_mano"], o.clase)))
            rec = next((r for r in self._portes if o.id in r["oids"]), None)
            if rec is None:          # ¿el mismo objeto con otro número? Cerca de donde estaba, o en manos de quien lo sujetaba
                cand = [r for r in self._portes if r["id"] not in vistos and 0 < t - r["ult"] <= 3.0
                        and r["clase"] == o.clase and (
                            float(np.linalg.norm(r["centro"] - o.centro)) <= 3.0 * max(r["diag"], diag)
                            or set(quienes) & set(r["muestras"][-1][1] if r["muestras"] else ()))]
                rec = min(cand, key=lambda r: float(np.linalg.norm(r["centro"] - o.centro))) if cand else None
            if rec is None:
                self._n_porte += 1
                rec = {"id": self._n_porte, "oids": set(), "clase": o.clase, "muestras": [], "dueno": None,
                       "forcejeo": [], "avisado": False}
                self._portes.append(rec)
            rec["oids"].add(o.id)
            vistos.add(rec["id"])
            rec.update(centro=o.centro.copy(), diag=diag, ult=t, caja=o.caja.copy())
            rec["muestras"] = [m for m in rec["muestras"] if t - m[0] <= 8.0] + [(t, quienes)]
            if len(quienes) >= 2:
                rec["forcejeo"] = [x for x in rec["forcejeo"] if t - x <= 6.0] + [t]
            if rec["dueno"] is None:                         # lo lleva una sola persona de forma sostenida
                recientes = [q for tm, q in rec["muestras"] if t - tm <= c["tiempo_porte"]]
                if rec["muestras"] and t - rec["muestras"][0][0] >= 0.9 * c["tiempo_porte"] and recientes:
                    solos = [q[0] for q in recientes if len(q) == 1]
                    if solos and solos.count(solos[-1]) >= 0.8 * len(recientes):
                        rec["dueno"], rec["dueno_desde"] = solos[-1], t

        for rec in self._portes:
            if rec["avisado"] or rec["dueno"] is None:
                continue
            a = rec["dueno"]
            # con pocos cuadros por segundo, «lo tiene el otro desde hace 0.6 s» puede ser un solo cuadro:
            # la ventana se alarga para que haya al menos dos
            paso = float(np.median(np.diff([tm for tm, _ in rec["muestras"]]))) if len(rec["muestras"]) > 1 else 0.1
            cambio = max(c["cambio_min"], 1.5 * paso)
            ultimos = [(tm, q) for tm, q in rec["muestras"] if rec["ult"] - tm <= cambio]
            if not ultimos or rec["ult"] - ultimos[0][0] < 0.6 * cambio:
                continue
            otros = [q for _, q in ultimos if q and a not in q]
            if len(otros) < 0.8 * len(ultimos):
                continue                                     # el dueño todavía lo tiene, o nadie lo lleva
            b = otros[-1][0]
            # pasó directo de las manos (o la espalda) del dueño a las de otro: no lo recogió del piso horas después
            con_dueno = [tm for tm, q in rec["muestras"] if a in q]
            if not con_dueno or ultimos[0][0] - con_dueno[-1] > max(2.0, 3 * paso):
                continue
            pa, pb = personas.get(a), personas.get(b)
            if pb is None and b in hist.personas and t - hist.personas[b].ultima_vez <= 1.0:
                pb = hist.personas[b].muestras[-1]
            if pa is None and a in hist.personas and t - hist.personas[a].ultima_vez <= 2.0:
                pa = hist.personas[a].muestras[-1]
            if pa is None or pb is None:
                continue
            ca = np.array([(pa.caja[0] + pa.caja[2]) / 2, (pa.caja[1] + pa.caja[3]) / 2])
            cb = np.array([(pb.caja[0] + pb.caja[2]) / 2, (pb.caja[1] + pb.caja[3]) / 2])
            alto = (alto_estimado(pa.caja, pa.puntos, pa.conf_puntos) + alto_estimado(pb.caja, pb.puntos, pb.conf_puntos)) / 2
            sep = float(np.linalg.norm(ca - cb)) / alto
            if sep < c["separacion"]:
                continue                                     # siguen juntos: puede ser que se la esté mostrando
            rec["avisado"] = True
            forcejeo = [x for x in rec["forcejeo"] if t - x <= 6.0]
            lucha = (f" Hubo forcejeo: ambas personas sujetaban la {rec['clase']} a la vez ({len(forcejeo)} cuadros)."
                     if forcejeo else "")
            out.append(Evento(
                "hurto", "alta" if forcejeo or (a, b) in self._contactos or (b, a) in self._contactos else "media",
                f"Posible arrebato: la persona #{b} le quitó una {rec['clase']} a la persona #{a}",
                f"La {rec['clase']} la llevaba encima la persona #{a} (en la mano o a la espalda) desde hace "
                f"{rec['ult'] - rec['dueno_desde'] + c['tiempo_porte']:.0f} s; ahora la lleva la persona #{b}, que ya está "
                f"a {sep:.1f} alturas de #{a} (umbral {c['separacion']}).{lucha}",
                f"arrebato:{rec['id']}:{b}", [rec["caja"].tolist(), list(map(float, pb.caja))]))
        self._portes = [r for r in self._portes if t - r["ult"] <= 6.0]
        return out

    # ---- robo de vehículos: alguien distinto de quien lo estacionó lo manipula o se lo lleva
    def _vehiculos(self, obs) -> list[Evento]:
        """Se recuerda cada vehículo estacionado por su lugar, no por su ID de seguimiento: cuando alguien
        se sube a una moto, el seguidor suele perderla y le da otro número."""
        c = self.cfg["vehiculos"]
        t = obs.t
        out = []
        self._puestos = [q for q in self._puestos if t - q["ult"] <= 5.0]
        for o in obs.objetos:
            if o.clase not in VEHICULOS or self._ignorado(o.caja):
                continue
            diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
            cerca = [q for q in self._puestos if q["clase"] == o.clase
                     and np.linalg.norm(q["centro"] - o.centro) < 0.6 * q["diag"]]
            if cerca:
                q = min(cerca, key=lambda x: np.linalg.norm(x["centro"] - o.centro))
            else:
                encima = [p for p in obs.personas if solape(caja_completa(p), o.caja) >= c["solape_min"]]
                q = {"clase": o.clase, "centro": o.centro.copy(), "diag": diag, "desde": t, "ult": t,
                     "dueno": encima[0].id if encima else None, "toques": {}}
                self._puestos.append(q)
                continue
            q["ult"] = t
            estacionado = t - q["desde"]
            movido = float(np.linalg.norm(o.centro - q["centro"])) / q["diag"]
            encima = {p.id: p for p in obs.personas if solape(caja_completa(p), o.caja) >= c["solape_min"]}
            for pid in list(q["toques"]):
                if pid not in encima:
                    del q["toques"][pid]
            for pid, p in encima.items():
                inicio = q["toques"].setdefault(pid, t)
                parado_antes = inicio - q["desde"]
                if pid == q["dueno"] or parado_antes < c["tiempo_estacionado"]:
                    continue
                dueno = (f"no es quien lo estacionó (persona #{q['dueno']})" if q["dueno"] is not None
                         else "no se vio a nadie estacionarlo")
                nombre = f"vehículo ({o.clase})"
                if movido > 0.35:
                    out.append(Evento(
                        "robo_vehiculo", "alta", f"Posible robo: la persona #{pid} se lleva un {nombre}",
                        f"El {nombre} llevaba {parado_antes:.0f} s estacionado; ahora se desplazó {movido:.1f} veces "
                        f"su tamaño con la persona #{pid} encima ({dueno}).",
                        f"robo_vehiculo_mov:{pid}:{int(q['desde'])}", [o.caja.tolist(), p.caja.tolist()]))
                elif t - inicio >= c["tiempo_manipulacion"]:
                    out.append(Evento(
                        "robo_vehiculo", "media", f"Posible robo: la persona #{pid} manipula un {nombre} estacionado",
                        f"El {nombre} llevaba {parado_antes:.0f} s estacionado y la persona #{pid} está sobre él o "
                        f"pegada a él desde hace {t - inicio:.0f} s; {dueno}.",
                        f"robo_vehiculo:{pid}:{int(q['desde'])}", [o.caja.tolist(), p.caja.tolist()]))
        return out

    # ---- robo con huida: alguien toca a otra persona y sale corriendo
    def _huida(self, obs, hist) -> list[Evento]:
        c = self.cfg["huida"]
        t = obs.t
        out = []
        act = {p.id: p for p in obs.personas if p.id in hist.personas}
        for (a, b), (tc, dc) in list(self._contactos.items()):   # los registra _registrar_contactos
            if t - tc > c["ventana"]:
                del self._contactos[(a, b)]
                continue
            if a not in act or a not in hist.personas or (a, b) not in self._contactos:
                continue
            va = velocidad_cuerpo(hist.personas[a], t)
            vb = velocidad_cuerpo(hist.personas[b], t) if b in hist.personas else 0.0
            if va >= c["velocidad"] and vb < c["velocidad"]:   # huye quien tocó; la víctima no corre con él
                out.append(Evento(
                    "robo_huida", "alta", f"Posible asalto o arrebato: la persona #{a} tocó a #{b} y huyó corriendo",
                    f"La mano o el pie de #{a} llegó al cuerpo de #{b} y {t - tc:.1f} s después #{a} se aleja a "
                    f"{va:.1f} alturas/s (umbral {c['velocidad']}); #{b} se mueve a {vb:.1f}.",
                    f"robo_huida:{a}:{b}", [act[a].caja.tolist()]))
                self._contactos.pop((a, b), None)
                self._contactos.pop((b, a), None)
                continue
            # Con una cámara que se mueve (un celular en la mano que sigue al que huye) la velocidad de cada persona
            # en la imagen engaña; la distancia entre las dos, no: si se separan muy rápido después del contacto,
            # una de ellas huyó, aunque la cámara la haya seguido.
            if b not in act or t - tc < 0.5:
                continue
            pa, pb = act[a], act[b]
            dn = float(np.linalg.norm(pa.centro - pb.centro)) / ((pa.alto + pb.alto) / 2)
            ritmo = (dn - dc) / (t - tc)
            if ritmo >= c.get("separacion", 1.8) and dn - dc >= 1.5:
                out.append(Evento(
                    "robo_huida", "alta", f"Posible asalto o arrebato: la persona #{a} tocó a #{b} y se separaron corriendo",
                    f"La mano o el pie de #{a} llegó al cuerpo de #{b}; en los {t - tc:.1f} s siguientes se alejaron "
                    f"a {ritmo:.1f} alturas/s (umbral {c.get('separacion', 1.8)}), de {dc:.1f} a {dn:.1f} alturas de "
                    "distancia. Esta medida no depende de que la cámara esté quieta.",
                    f"robo_huida:{a}:{b}", [pa.caja.tolist(), pb.caja.tolist()]))
                self._contactos.pop((a, b), None)
                self._contactos.pop((b, a), None)
        return out

    # ---- vandalismo: golpes rápidos contra un vehículo o una zona protegida, sin otra persona cerca
    def _vandalismo(self, obs, hist, ancho, alto, zonas) -> list[Evento]:
        cv = self.cfg["violencia"]
        t = obs.t
        out = []
        protegidas = [z for z in zonas if z.get("tipo") == "protegida" and zona_activa(z, self.reloj())]
        vehiculos = [o for o in obs.objetos if o.clase in VEHICULOS]
        for p in obs.personas:
            pista = hist.personas.get(p.id)
            dur, vmax = self._rap.get(p.id, (0.0, 0.0))
            if pista is None or dur < cv["duracion_min"]:
                continue
            otros = [q for q in obs.personas if q.id != p.id and
                     float(np.linalg.norm(q.centro - p.centro)) / p.alto <= 1.5]
            if otros:                                   # con alguien cerca, es la regla de violencia
                continue
            for o in vehiculos:
                x1, y1, x2, y2 = o.caja
                if x1 <= p.centro[0] <= x2 and y1 <= p.centro[1] <= y2:
                    continue                            # está sobre el vehículo: lo cubre la regla de robo
                if extremidad_en(pista, t, lambda x, y: x1 <= x <= x2 and y1 <= y <= y2):
                    out.append(Evento(
                        "vandalismo", "alta", f"Posible vandalismo: la persona #{p.id} golpea un vehículo ({o.clase})",
                        f"Extremidades a {vmax:.1f} alturas/s durante {dur:.1f} s que llegan al vehículo, sin otra "
                        "persona cerca.", f"vandalismo:{p.id}:{o.id}", [p.caja.tolist(), o.caja.tolist()]))
                    break
            for z in protegidas:
                if extremidad_en(pista, t, lambda x, y: punto_en_poligono(x / ancho, y / alto, z["puntos"])):
                    out.append(Evento(
                        "vandalismo", "alta", f"Posible vandalismo en «{z['nombre']}» (persona #{p.id})",
                        f"Extremidades a {vmax:.1f} alturas/s durante {dur:.1f} s dentro de la zona protegida, "
                        "sin otra persona cerca.", f"vandalismo:{z['id']}:{p.id}", [p.caja.tolist()]))
        return out

    # ---- armas: cuchillo o bate en la mano de una persona
    def _armas(self, obs, hist) -> list[Evento]:
        c = self.cfg["armas"]
        t = obs.t
        out = []
        for o in obs.objetos:
            if o.clase not in ARMAS:
                continue
            pista_o = hist.objetos.get(o.id)
            if pista_o is None or len(pista_o.ventana(t, 2.0)) < c["detecciones_min"]:
                continue                                # una sola detección suele ser un error
            for p in obs.personas:
                manos = [p.punto(k) for k in (MUNECA_I, MUNECA_D)]
                d = min((float(np.linalg.norm(m - o.centro)) for m in manos if m is not None), default=1e9)
                if d / p.alto <= c["distancia_mano"]:
                    out.append(Evento(
                        "arma", "alta", f"Posible arma: la persona #{p.id} sostiene un {o.clase}",
                        f"Se detectó un {o.clase} {len(pista_o.ventana(t, 2.0))} veces en 2 s, a {d / p.alto:.2f} "
                        f"alturas de la mano (límite {c['distancia_mano']}). Verifique la imagen: a distancia, "
                        "el detector confunde objetos alargados.", f"arma:{p.id}:{o.clase}",
                        [p.caja.tolist(), o.caja.tolist()]))
                    break
        return out

    # ---- aglomeración y huida masiva
    def _aglomeracion(self, obs, hist) -> list[Evento]:
        c = self.cfg["aglomeracion"]
        t = obs.t
        out = []
        # personas vistas en el último segundo: en una multitud el detector pierde a varias en cada cuadro
        n = max(len(obs.personas), sum(1 for q in hist.personas.values() if t - q.ultima_vez <= 1.0))
        if n >= c["max_personas"]:
            self._multitud_bajo = None
            if self._multitud_desde is None:
                self._multitud_desde = t
            if t - self._multitud_desde >= c["duracion"]:
                out.append(Evento(
                    "aglomeracion", "media", f"Aglomeración: {n} personas en la cámara",
                    f"Hay {n} personas o más (límite {c['max_personas']}) desde hace {t - self._multitud_desde:.0f} s.",
                    "aglomeracion"))
        elif self._multitud_desde is not None:
            # unos cuadros con menos personas (tapadas, perdidas) no cortan la aglomeración: 2 s seguidos sí
            if getattr(self, "_multitud_bajo", None) is None:
                self._multitud_bajo = t
            if t - self._multitud_bajo >= 2.0:
                self._multitud_desde = self._multitud_bajo = None
        vel_huida = self.cfg.get("huida", {}).get("velocidad", 1.6)
        corriendo = [p for p in obs.personas if p.id in hist.personas
                     and velocidad_cuerpo(hist.personas[p.id], t) >= vel_huida]
        if len(corriendo) >= c["min_corriendo"] and len(corriendo) >= 0.5 * n:
            out.append(Evento(
                "estampida", "alta", f"Huida masiva: {len(corriendo)} de {n} personas corren a la vez",
                f"{len(corriendo)} personas se mueven a más de {vel_huida} alturas/s al mismo tiempo; puede indicar "
                "una amenaza que la cámara no ve.", "estampida", [p.caja.tolist() for p in corriendo]))
        return out

    # ---- emociones (solo informativas, salvo que se active la alerta)
    def _emociones(self, obs, hist) -> list[Evento]:
        c = self.cfg["emociones"]
        t = obs.t
        out = []
        for r in obs.rostros:
            if r.persona_id is None or r.probs is None or r.persona_id not in hist.personas:
                continue
            est = hist.personas[r.persona_id].estado
            prev = est.get("emo")
            est["emo"] = r.probs if prev is None else 0.7 * prev + 0.3 * r.probs
            etiqueta = EMOCIONES[int(np.argmax(est["emo"]))]
            if etiqueta != est.get("emo_etiqueta"):
                est["emo_etiqueta"], est["emo_desde"] = etiqueta, t
            if c["alerta_enojo_miedo"] and etiqueta in ("enojo", "miedo") and \
                    t - est["emo_desde"] >= c["duracion_min"]:
                out.append(Evento(
                    "emocion", "baja", f"Persona #{r.persona_id} con expresión de {etiqueta} sostenida",
                    f"Expresión predominante «{etiqueta}» durante {t - est['emo_desde']:.0f} s "
                    f"(probabilidad {float(np.max(est['emo'])):.0%}).",
                    f"emocion:{r.persona_id}:{etiqueta}"))
        return out
