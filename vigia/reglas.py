"""Reglas de comportamiento sobre la pose y el seguimiento.

Cada regla es transparente: trabaja con medidas físicas (velocidad de las extremidades,
ángulo del tronco, tiempo en una zona) y devuelve la evidencia numérica que la disparó.
Así cada alerta se puede explicar, que es uno de los vacíos que señaló la revisión.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .historial import Historial, Muestra, Pista
from .tipos import (CADERA_D, CADERA_I, CODO_D, CODO_I, EMOCIONES, HOMBRO_D, HOMBRO_I,
                    MUNECA_D, MUNECA_I, TOBILLO_D, TOBILLO_I, Evento, Observacion)

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

    def actualizar_cfg(self, cfg: dict):
        self.cfg = cfg

    def evaluar(self, obs: Observacion, hist: Historial, ancho: int, alto: int,
                zonas: list[dict]) -> list[Evento]:
        ev: list[Evento] = []
        r = self.cfg
        if r["violencia"]["activa"]:
            ev += self._violencia(obs, hist)
        if r["caida"]["activa"]:
            ev += self._caida(obs, hist)
        if r["merodeo"]["activa"] or r["intrusion"]["activa"]:
            ev += self._zonas(obs, hist, ancho, alto, zonas)
        if r["objetos"]["activa"]:
            ev += self._objetos(obs, hist)
        ev += self._emociones(obs, hist)
        return ev

    # ---- violencia: dos personas cerca con extremidades muy rápidas
    def _violencia(self, obs, hist) -> list[Evento]:
        c = self.cfg["violencia"]
        t = obs.t
        pistas = {p.id: hist.personas[p.id] for p in obs.personas if p.id in hist.personas}
        actuales = {p.id: p for p in obs.personas}
        rapidos = {}
        for pid, pista in pistas.items():
            alto = pista.alto_de_pie(t)
            vel = velocidades_extremidades(pista, t, 1.5, alto)
            # tiempo acumulado sobre el umbral: cada muestra rápida aporta el intervalo típico entre frames
            paso = float(np.median(np.diff([ti for ti, _ in vel]))) if len(vel) > 1 else 0.125
            dur = paso * sum(1 for _, v in vel if v >= c["velocidad_extremidades"])
            pista.estado["vel_max"] = max((v for _, v in vel), default=0.0)
            rapidos[pid] = (dur, pista.estado["vel_max"])
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
                if dur >= c["duracion_min"]:
                    eventos.append(Evento(
                        "violencia", "alta", f"Posible pelea entre las personas #{a} y #{b}",
                        f"Extremidades a {vmax:.1f} alturas/s (umbral {c['velocidad_extremidades']}) durante "
                        f"{dur:.1f} s, con {dist:.2f} alturas de separación.",
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
                        "Los pies de la persona están dentro del polígono de la zona restringida.",
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
            if pista is None:
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
