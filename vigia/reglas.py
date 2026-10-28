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
                    MUNECA_D, MUNECA_I, TOBILLO_D, TOBILLO_I, Evento, Observacion)

VEHICULOS = {"bicicleta", "auto", "motocicleta"}
PORTATILES = {"mochila", "bolso", "maleta", "laptop", "celular"}
ARMAS = {"cuchillo", "bate"}

EXTREMIDADES = [MUNECA_I, MUNECA_D, CODO_I, CODO_D, TOBILLO_I, TOBILLO_D]


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


# --------------------------------------------------------------------------- reglas
class Reglas:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._puestos: list[dict] = []   # vehículos estacionados vistos por esta cámara
        self._contactos: dict = {}       # (quien toca, a quien) -> último instante de contacto
        self._multitud_desde = None
        self.reloj = lambda: datetime.now().time()   # se reemplaza en las pruebas

    def actualizar_cfg(self, cfg: dict):
        self.cfg = cfg

    def evaluar(self, obs: Observacion, hist: Historial, ancho: int, alto: int,
                zonas: list[dict]) -> list[Evento]:
        ev: list[Evento] = []
        r = self.cfg
        self._rap = self._rapidez(obs, hist)
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
            out[p.id] = (dur, pista.estado["vel_max"])
        return out

    # ---- violencia: dos personas cerca con extremidades muy rápidas y contacto
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
                if dur < c["duracion_min"]:
                    continue
                if c.get("requiere_contacto", True) and not (
                        contacto(pistas[a], pb, t) or contacto(pistas[b], pa, t)):
                    continue
                eventos.append(Evento(
                    "violencia", "alta", f"Posible pelea entre las personas #{a} y #{b}",
                    f"Extremidades a {vmax:.1f} alturas/s (umbral {c['velocidad_extremidades']}) durante "
                    f"{dur:.1f} s, con {dist:.2f} alturas de separación y contacto de una mano o un pie "
                    "con el tronco o la cabeza de la otra persona.",
                    f"violencia:{a}-{b}", [pa.caja.tolist(), pb.caja.tolist()]))
        return eventos

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
            if es_horizontal(ultima, c["angulo_horizontal"]):
                if est["horiz_desde"] is None:
                    est["horiz_desde"] = t
                est["vert_desde"] = None
            else:
                if est["vert_desde"] is None:
                    est["vert_desde"] = t
                if t - est["vert_desde"] >= 1.0:   # se levantó: se reinicia el episodio
                    est.update(horiz_desde=None, caida=False, suelo=False)
                continue
            tendido = t - est["horiz_desde"]
            alto = pista.alto_de_pie(t, 6.0)
            if not est["caida"] and tendido >= 0.5:
                previas = [m for m in pista.muestras if est["horiz_desde"] - 2.5 <= m.t <= est["horiz_desde"]]
                if previas:
                    ys = [_torso(m)[1] for m in previas]
                    descenso = (_torso(ultima)[1] - min(ys)) / alto
                    if descenso >= c["descenso_min"]:
                        est["caida"] = True
                        out.append(Evento(
                            "caida", "alta", f"Caída de la persona #{p.id}",
                            f"El centro de la cadera bajó {descenso:.2f} alturas en menos de 2.5 s y el "
                            f"tronco quedó a más de {c['angulo_horizontal']}° de la vertical.",
                            f"caida:{p.id}", [p.caja.tolist()]))
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
            fx, fy = p.pies[0] / ancho, p.pies[1] / alto
            desde = pista.estado.setdefault("zona_desde", {})
            for z in zonas:
                dentro = punto_en_poligono(fx, fy, z["puntos"])
                if not dentro:
                    desde.pop(z["id"], None)
                    continue
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
            if not zonas_merodeo and c["merodeo"]["activa"]:
                estancia = t - pista.primera_vez
                if estancia >= c["merodeo"]["tiempo_max"]:
                    out.append(Evento(
                        "merodeo", "media", f"Merodeo: persona #{p.id} permanece en la escena",
                        f"Visible durante {estancia:.0f} s (límite {c['merodeo']['tiempo_max']} s; "
                        "sin zonas de merodeo definidas, se mide en toda la cámara).",
                        f"merodeo:cam:{p.id}", [p.caja.tolist()]))
        return out

    # ---- objetos abandonados y posible hurto
    def _objetos(self, obs, hist) -> list[Evento]:
        c = self.cfg["objetos"]
        t = obs.t
        out = []
        for o in obs.objetos:
            pista = hist.objetos.get(o.id)
            if pista is None or o.clase not in PORTATILES:   # vehículos y armas tienen sus propias reglas
                continue
            est = pista.estado
            diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
            ultimos = pista.ventana(t, 1.5)
            mov = float(np.linalg.norm(ultimos[-1].caja[:2] - ultimos[0].caja[:2])) if len(ultimos) > 1 else 0.0
            quieto = mov < 0.35 * diag

            def cercania(per):
                return float(np.linalg.norm(per.centro - o.centro)) / per.alto

            cerca = sorted(((cercania(per), per) for per in obs.personas), key=lambda x: x[0])
            if "dueno" not in est and cerca and cerca[0][0] <= 1.0:
                est["dueno"] = cerca[0][1].id
            if quieto:
                est.setdefault("quieto_desde", t)
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

    # ---- robo de vehículos: alguien distinto de quien lo estacionó lo manipula o se lo lleva
    def _vehiculos(self, obs) -> list[Evento]:
        """Se recuerda cada vehículo estacionado por su lugar, no por su ID de seguimiento: cuando alguien
        se sube a una moto, el seguidor suele perderla y le da otro número."""
        c = self.cfg["vehiculos"]
        t = obs.t
        out = []
        self._puestos = [q for q in self._puestos if t - q["ult"] <= 5.0]
        for o in obs.objetos:
            if o.clase not in VEHICULOS:
                continue
            diag = float(np.hypot(o.caja[2] - o.caja[0], o.caja[3] - o.caja[1]))
            cerca = [q for q in self._puestos if q["clase"] == o.clase
                     and np.linalg.norm(q["centro"] - o.centro) < 0.6 * q["diag"]]
            if cerca:
                q = min(cerca, key=lambda x: np.linalg.norm(x["centro"] - o.centro))
            else:
                encima = [p for p in obs.personas if solape(p.caja, o.caja) >= c["solape_min"]]
                q = {"clase": o.clase, "centro": o.centro.copy(), "diag": diag, "desde": t, "ult": t,
                     "dueno": encima[0].id if encima else None, "toques": {}}
                self._puestos.append(q)
                continue
            q["ult"] = t
            estacionado = t - q["desde"]
            movido = float(np.linalg.norm(o.centro - q["centro"])) / q["diag"]
            encima = {p.id: p for p in obs.personas if solape(p.caja, o.caja) >= c["solape_min"]}
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
        for a, pa in act.items():                       # registrar contactos recientes
            for b, pb in act.items():
                if a == b:
                    continue
                dist = float(np.linalg.norm(pa.centro - pb.centro)) / ((pa.alto + pb.alto) / 2)
                if dist <= 1.2 and contacto(hist.personas[a], pb, t, ventana=0.5):
                    self._contactos[(a, b)] = t
        for (a, b), tc in list(self._contactos.items()):
            if t - tc > c["ventana"]:
                del self._contactos[(a, b)]
                continue
            if a not in act or a not in hist.personas:
                continue
            va = velocidad_cuerpo(hist.personas[a], t)
            vb = velocidad_cuerpo(hist.personas[b], t) if b in hist.personas else 0.0
            if va >= c["velocidad"] and vb < c["velocidad"]:   # huye quien tocó; la víctima no corre con él
                out.append(Evento(
                    "robo_huida", "alta", f"Posible asalto o arrebato: la persona #{a} tocó a #{b} y huyó corriendo",
                    f"La mano o el pie de #{a} llegó al cuerpo de #{b} y {t - tc:.1f} s después #{a} se aleja a "
                    f"{va:.1f} alturas/s (umbral {c['velocidad']}); #{b} se mueve a {vb:.1f}.",
                    f"robo_huida:{a}:{b}", [act[a].caja.tolist()]))
                del self._contactos[(a, b)]
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
        n = len(obs.personas)
        if n >= c["max_personas"]:
            if self._multitud_desde is None:
                self._multitud_desde = t
            if t - self._multitud_desde >= c["duracion"]:
                out.append(Evento(
                    "aglomeracion", "media", f"Aglomeración: {n} personas en la cámara",
                    f"Hay {n} personas o más (límite {c['max_personas']}) desde hace {t - self._multitud_desde:.0f} s.",
                    "aglomeracion"))
        else:
            self._multitud_desde = None
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
