"""Banco de estrés: cada situación que VigIA debe detectar, pasada por las condiciones reales de una prueba en
vivo (PC lenta, números de seguimiento que cambian, pies tapados, puntos con ruido, detecciones perdidas, cámara
temblorosa). Sirve para saber qué regla falla y con qué, antes de que pase en la feria.

  python pruebas/estres.py            imprime la tabla
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "pruebas"))

import test_reglas as T  # noqa: E402
from vigia.historial import Historial  # noqa: E402
from vigia.objetos import FiltroObjetos  # noqa: E402
from vigia.reglas import Reglas  # noqa: E402
from vigia.simulador import ALTO, ANCHO, H, ZONAS_DEMO, Simulador, esqueleto  # noqa: E402
from vigia.tipos import Objeto, Observacion, Persona  # noqa: E402

# ------------------------------------------------------------------ situaciones
def demo():
    sim = Simulador()
    return lambda t: (lambda o: (o.personas, o.objetos))(sim.frame(t)[1])


def forcejeo(t):
    a = esqueleto(400 + 0.06 * H * math.sin(2 * math.pi * 1.2 * t), 480, "quieto", t, 0, 1, 0)
    b = esqueleto(400 + 0.08 * H - 0.06 * H * math.sin(2 * math.pi * 1.2 * t), 480, "quieto", t, 0, -1, 0)
    return [T._con_munecas_tapadas(1, a), T._con_munecas_tapadas(2, b)], []


def golpes(t):
    a = esqueleto(400, 480, "golpea", t, 0, 1, 0)
    b = esqueleto(400 + 0.55 * H, 480, "quieto", t, 0, -1, 0)
    return [T.persona(1, a), T.persona(2, b)], []


def huida(t):
    if t < 1.5:
        a = esqueleto(400, 480, "golpea", t, 0, 1, 0)
    else:
        x = 400 - 2.5 * H * (t - 1.5)
        a = esqueleto(x, 480, "camina", t, abs(x - 400), -1, 0)
    v = esqueleto(400 + 0.6 * H, 480, "quieto", t, 0, -1, 0)
    return [T.persona(1, a), T.persona(2, v)], []


def vandalismo(t):
    auto = np.array([400 + 0.35 * H, 480 - 1.0 * H, 400 + 1.6 * H, 480.0])
    return [T.persona(1, esqueleto(400, 480, "golpea", t, 0, 1, 0))], [Objeto(9, "auto", auto)]


def arma(t):
    k = esqueleto(400, 480, "quieto", 0, 0, 1, 0)
    w = k[9]
    return [T.persona(1, k)], [Objeto(5, "cuchillo", np.array([w[0] - 8, w[1] - 20, w[0] + 8, w[1] + 5]), 0.6)]


def moto(t):
    m = np.array([300, 380, 520, 480.0])
    if t > 11:
        m = m + [(t - 11) * 60, 0, (t - 11) * 60, 0]
    k = esqueleto(410, 470, "quieto", 0, 0, 1, 0)
    caja = np.array([370, 240, 450, 472.0])
    ps = []
    if t < 1:
        ps.append(Persona(1, caja, k, np.full(17, 0.9)))
    if t >= 7:
        ps.append(Persona(2, caja + (m - np.array([300, 380, 520, 480.0]))[[0, 1, 0, 1]], k, np.full(17, 0.9)))
    return ps, [Objeto(9, "motocicleta", m)]


def multitud(n, vel):
    def f(t):
        ps = []
        for i in range(n):
            x = 60 + i * 55 + vel * H * t
            ps.append(T.persona(i + 1, esqueleto(x, 300 + (i % 3) * 90, "camina" if vel else "quieto", t, vel * H * t, 1, 0)))
        return ps, []
    return f


def caida_sola(t):
    prog = min(1.0, max(0.0, (t - 3.0) / 0.7))
    b = esqueleto(400, 480, "quieto" if t < 3 else ("cae" if prog < 1 else "tendido"), t, 0, 1, prog)
    return [T.persona(2, b)], []


def _mochila_en_mano(k, H_=None):
    """Caja de una mochila agarrada por la muñeca derecha del esqueleto k."""
    w = k[10]
    return np.array([w[0] - 0.09 * H, w[1] - 0.05 * H, w[0] + 0.09 * H, w[1] + 0.22 * H])


def arrebato_sale(t):
    """Como el video de prueba de la feria: A tiene la mochila; B llega corriendo, forcejean 1.5 s, B se la queda y
    sale del cuadro en medio segundo. A se queda sin ella."""
    a = esqueleto(300, 470, "quieto", t, 0, 1, 0)
    ps, objs = [T.persona(1, a)], []
    if t < 6.0:
        objs.append(Objeto(40, "mochila", _mochila_en_mano(a)))
    if 5.0 <= t < 9.2:
        if t < 6.0:
            xb = 900 - (t - 5.0) * 500
        elif t < 8.5:
            xb = 400 + 0.05 * H * math.sin(9 * t)
        else:
            xb = 400 + (t - 8.5) * 2.8 * H
        b = esqueleto(xb, 470, "camina" if (t < 6 or t >= 8.5) else "quieto", t, xb, -1 if t < 6 else 1, 0)
        ps.append(T.persona(2, b))
        if 6.0 <= t < 8.5:                          # forcejeo: los dos sujetan la mochila entre ellos
            objs.append(Objeto(40, "mochila", np.array([330, 380, 400, 450.0])))
        elif t >= 8.5 and xb < ANCHO - 30:
            objs.append(Objeto(41, "mochila", _mochila_en_mano(b)))
    return ps, objs


def toca_y_sale(t):
    """A está parado; B llega caminando, le toca el hombro y sale corriendo del cuadro."""
    a = esqueleto(500, 470, "quieto", t, 0, 1, 0)
    ps = [T.persona(1, a)]
    if 1.0 <= t < 3.0:
        xb = 1000 - (t - 1.0) * 200
    elif t < 3.6:
        xb = 600.0
    else:
        xb = 600 + (t - 3.6) * 3.0 * H
    if 1.0 <= t and xb < ANCHO + 20:
        b = esqueleto(xb, 470, "camina" if t < 3.0 or t >= 3.6 else "quieto", t, xb, -1 if t < 3.6 else 1, 0)
        if 3.0 <= t < 3.6:                          # la mano sube al hombro de A en 0.3 s y se queda ahí
            f = min(1.0, (t - 3.0) / 0.3)
            b[10] = (1 - f) * b[10] + f * (a[6] + np.array([10, 5]))
        ps.append(T.persona(2, b))
    return ps, []


SITUACIONES = [
    # nombre, generador (fábrica), segundos, tipo esperado, zonas, filtro de mensaje
    ("merodeo", demo, 40, "merodeo", ZONAS_DEMO, None),
    ("intrusión", demo, 40, "intrusion", ZONAS_DEMO, None),
    ("pelea (golpes)", lambda: golpes, 4, "violencia", (), None),
    ("forcejeo", lambda: forcejeo, 5, "violencia", (), None),
    ("caída", lambda: caida_sola, 10, "caida", (), None),
    ("en el suelo", lambda: caida_sola, 12, "en_suelo", (), None),
    ("abandono", demo, 88, "abandono", ZONAS_DEMO, None),
    ("hurto (del piso)", demo, 100, "hurto", ZONAS_DEMO, None),
    ("arrebato", lambda: T.Arrebato().cuadros(T.Arrebato.tiron), 9, "hurto", (), "arrebato"),
    ("robo con huida", lambda: huida, 4, "robo_huida", (), None),
    ("robo de vehículo", lambda: moto, 14, "robo_vehiculo", (), None),
    ("vandalismo", lambda: vandalismo, 3, "vandalismo", (), None),
    ("arma", lambda: arma, 5, "arma", (), None),
    ("aglomeración", lambda: multitud(16, 0.0), 7, "aglomeracion", (), None),
    ("huida masiva", lambda: multitud(5, 2.5), 6.5, "estampida", (), None),
    ("arrebato y sale", lambda: arrebato_sale, 11, "hurto", (), "arrebato"),
    ("toca y sale", lambda: toca_y_sale, 6, "robo_huida", (), None),
]


def abrazo(t):
    a = esqueleto(400, 480, "quieto", t, 0, 1, 0)
    b = esqueleto(400 + 0.08 * H + 0.004 * H * math.sin(9 * t), 480, "quieto", t, 0, -1, 0)
    return [T._con_munecas_tapadas(1, a), T._con_munecas_tapadas(2, b)], []


def caminar_juntos(t):
    x = 200 + 0.7 * H * t
    return [T.persona(1, esqueleto(x, 480, "camina", t, x, 1, 0)),
            T.persona(2, esqueleto(x + 0.1 * H, 485, "camina", t, x, 1, 0))], []


def conversacion(t):
    """Dos personas conversan a medio metro moviendo las manos (gestos), sin tocarse."""
    out = []
    for pid, x, d in ((1, 400, 1), (2, 400 + 0.75 * H, -1)):
        k = esqueleto(x, 480, "quieto", t, 0, d, 0)
        for w, e, fase in ((9, 7, 0.0), (10, 8, 1.3)):
            ang = 0.9 * math.sin(2 * math.pi * 0.8 * t + fase + pid)
            k[w] = k[e] + np.array([d * 0.12 * H * math.cos(ang), -0.12 * H * math.sin(ang)])
        out.append(T.persona(pid, k))
    return out, []


def mochila_puesta(t):
    """Alguien espera 40 s parado con la mochila a la espalda."""
    k = esqueleto(400, 480, "quieto", t, 0, 1, 0)
    return [T.persona(1, k)], [Objeto(70, "mochila", np.array([400 - 0.10 * H, 480 - 0.78 * H, 400 + 0.03 * H, 480 - 0.50 * H]))]


def dueno_recoge(t):
    """Deja su mochila, se aleja 15 s y vuelve por ella."""
    bolsa = np.array([280, 440, 330, 480.0])
    if t < 3:
        x = 300.0
    elif t < 8:
        x = 300 + (t - 3) * 90
    elif t < 18:
        x = 750.0
    elif t < 22:
        x = 750 - (t - 18) * 110
    else:
        x = 310 - (t - 22) * 120
    ps = [T.persona(1, esqueleto(x, 480, "camina" if 3 <= t < 8 or t >= 18 else "quieto", t, x, 1 if t < 18 else -1, 0))]
    if t < 22:
        objs = [Objeto(50, "mochila", bolsa)] if t >= 2 else []
    else:
        objs = [Objeto(51, "mochila", np.array([x - 25, 360, x + 25, 400.0]))]
    return ps, objs


def paso_por_zona(t):
    x = 60 + 0.8 * H * t
    return [T.persona(1, esqueleto(x, 470, "camina", t, x, 1, 0))], []


def grupo_caminando(t):
    return multitud(10, 0.8)(t)


def saluda_y_se_va(t):
    """B le toca el hombro a A (un saludo) y se va caminando del cuadro."""
    ps, objs = toca_y_sale(t)
    if t >= 3.6 and len(ps) > 1:
        xb = 600 + (t - 3.6) * 0.6 * H
        ps[1] = T.persona(2, esqueleto(xb, 470, "camina", t, xb, 1, 0)) if xb < ANCHO + 20 else ps[1]
        if xb >= ANCHO + 20:
            ps = ps[:1]
    return ps, objs


def tiron_fallido(t):
    """B intenta quitarle la mochila a A, no puede y se va corriendo sin ella."""
    ps, objs = arrebato_sale(t)
    if t >= 8.5:
        objs = [Objeto(40, "mochila", _mochila_en_mano(ps[0].puntos))]
    return ps, objs


NEGATIVOS = [
    # nombre, generador, segundos, tipos que NO deben salir, zonas
    ("cruzarse caminando", demo, 14, {"violencia", "robo_huida", "hurto"}, ZONAS_DEMO),
    ("abrazo", lambda: abrazo, 6, {"violencia"}, ()),
    ("caminar juntos", lambda: caminar_juntos, 4, {"violencia", "robo_huida"}, ()),
    ("conversar con gestos", lambda: conversacion, 8, {"violencia"}, ()),
    ("mochila puesta 40 s", lambda: mochila_puesta, 40, {"abandono", "hurto"}, ()),
    ("dueño recoge su mochila", lambda: dueno_recoge, 26, {"hurto"}, ()),
    ("pasar por una zona", lambda: paso_por_zona, 6, {"merodeo"}, ZONAS_DEMO),
    ("grupo caminando", lambda: grupo_caminando, 8, {"aglomeracion", "estampida", "violencia"}, ()),
    ("saluda y se va caminando", lambda: saluda_y_se_va, 8, {"robo_huida", "hurto"}, ()),
    ("tirón fallido", lambda: tiron_fallido, 11, {"hurto"}, ()),
]


# Límites conocidos: quien arrebata o toca y sale del cuadro en menos de un segundo se ve en 1 o 2 cuadros si la PC
# analiza menos de 5 por segundo. En vivo lo cubre la segunda mirada y un video subido se analiza en modo Detallado,
# los dos a 10 cuadros por segundo; la tabla los marca como «2ª mirada».
LIMITES = {("arrebato y sale", "PC lenta"), ("toca y sale", "PC lenta")}


def es_limite(situacion: str, condicion: str) -> bool:
    return any(situacion == s and condicion.startswith(c) for s, c in LIMITES)


# ------------------------------------------------------------------ condiciones reales
def _copiar(p: Persona, **cambios) -> Persona:
    d = dict(id=p.id, caja=p.caja.copy(), puntos=None if p.puntos is None else p.puntos.copy(),
             conf_puntos=None if p.conf_puntos is None else p.conf_puntos.copy(), conf=p.conf)
    d.update(cambios)
    return Persona(d["id"], np.asarray(d["caja"], float), d["puntos"], d["conf_puntos"], d["conf"])


class Condicion:
    paso = 0.1
    nombre = "ideal"

    def __init__(self, semilla=0):
        self.rng = np.random.default_rng(semilla)

    def aplicar(self, t, ps, objs):
        return ps, objs


class Lenta(Condicion):
    def __init__(self, paso, semilla=0):
        super().__init__(semilla)
        self.paso, self.nombre = paso, f"PC lenta ({1 / paso:.1f} c/s)"


class CambiaID(Condicion):
    """El seguidor le cambia el número a cada persona cada 4 s (pasa al cruzarse o taparse)."""
    nombre = "cambia el número"

    def aplicar(self, t, ps, objs):
        return [_copiar(p, id=p.id + 1000 * int(t // 4)) for p in ps], objs


class PiesTapados(Condicion):
    """Una mesa o el borde de la imagen tapa las piernas (como en el aula): sin rodillas ni tobillos."""
    nombre = "pies tapados"

    def aplicar(self, t, ps, objs):
        out = []
        for p in ps:
            c = p.conf_puntos.copy() if p.conf_puntos is not None else np.full(17, 0.9)
            c[13:17] = 0.05
            caja = p.caja.copy()
            caja[3] = caja[1] + 0.72 * (caja[3] - caja[1])
            out.append(_copiar(p, caja=caja, conf_puntos=c))
        return out, objs


class Ruido(Condicion):
    """Puntos del esqueleto y cajas con temblor (detector real, compresión del video)."""
    nombre = "ruido en los puntos"

    def aplicar(self, t, ps, objs):
        out = []
        for p in ps:
            alto = p.caja[3] - p.caja[1]
            pts = None if p.puntos is None else p.puntos + self.rng.normal(0, 0.015 * alto, p.puntos.shape)
            out.append(_copiar(p, caja=p.caja + self.rng.normal(0, 0.01 * alto, 4), puntos=pts))
        return out, objs


class Huecos(Condicion):
    """El detector pierde a cada persona en 20 % de los cuadros y a los objetos en 50 %."""
    nombre = "detecciones perdidas"

    def aplicar(self, t, ps, objs):
        return ([p for p in ps if self.rng.random() > 0.2],
                [Objeto(o.id, o.clase, o.caja, 0.5) for o in objs if self.rng.random() > 0.5])


class Temblor(Condicion):
    """Cámara en la mano: toda la imagen se desplaza unos píxeles en cada cuadro."""
    nombre = "cámara en la mano"

    def __init__(self, semilla=0):
        super().__init__(semilla)
        self.d = np.zeros(2)

    def aplicar(self, t, ps, objs):
        self.d = 0.85 * self.d + self.rng.normal(0, 6, 2)
        dx, dy = self.d
        mov = np.array([dx, dy, dx, dy])
        return ([_copiar(p, caja=p.caja + mov, puntos=None if p.puntos is None else p.puntos + self.d) for p in ps],
                [Objeto(o.id, o.clase, o.caja + mov, o.conf) for o in objs])


CONDICIONES = [lambda: Condicion(), lambda: Lenta(0.4), lambda: Lenta(0.7), CambiaID, PiesTapados, Ruido, Huecos,
               Temblor]


def correr(fabrica, segundos, zonas, condicion, cfg_reglas):
    gen = fabrica()
    reglas, hist = Reglas(copy.deepcopy(cfg_reglas)), Historial()
    filtro = FiltroObjetos(abierto=True)
    tipos, t = [], 0.0
    while t < segundos:
        ps, objs = gen(round(t, 2))
        ps, objs = condicion.aplicar(t, list(ps), list(objs))
        objs = filtro.filtrar(objs, t, cfg_reglas["objetos"], ps)
        obs = Observacion(t=t, personas=ps, objetos=objs)
        hist.actualizar(obs)
        tipos += reglas.evaluar(obs, hist, ANCHO, ALTO, list(zonas))
        t += condicion.paso
    return tipos


def tabla(situaciones=SITUACIONES, condiciones=CONDICIONES):
    cfg = copy.deepcopy(T.CFG["reglas"])
    cfg["merodeo"]["tiempo_max"] = 12
    cfg["objetos"]["tiempo_abandono"] = 10
    res = {}
    for nombre, fab, seg, tipo, zonas, texto in situaciones:
        for cf in condiciones:
            c = cf()
            ev = correr(fab, seg, zonas, c, cfg)
            ok = any(e.tipo == tipo and (texto is None or texto in e.mensaje) for e in ev)
            res[(nombre, c.nombre)] = ok
    return res


def falsas(negativos=NEGATIVOS, condiciones=CONDICIONES):
    cfg = copy.deepcopy(T.CFG["reglas"])
    cfg["merodeo"]["tiempo_max"] = 12
    cfg["objetos"]["tiempo_abandono"] = 10
    res = {}
    for nombre, fab, seg, prohibidos, zonas in negativos:
        for cf in condiciones:
            c = cf()
            ev = correr(fab, seg, zonas, c, cfg)
            res[(nombre, c.nombre)] = sorted({e.tipo for e in ev} & prohibidos)
    return res


if __name__ == "__main__":
    r = tabla()
    conds = list(dict.fromkeys(c for _, c in r))
    print(f"{'situación':20s} " + " | ".join(f"{c[:14]:14s}" for c in conds))
    for s in dict.fromkeys(s for s, _ in r):
        print(f"{s:20s} " + " | ".join(
            f"{'sí' if r[(s, c)] else ('2ª mirada' if es_limite(s, c) else 'NO'):14s}" for c in conds))
    print("\nFalsas alarmas (debe estar todo vacío):")
    f = falsas()
    for s in dict.fromkeys(s for s, _ in f):
        print(f"{s:24s} " + " | ".join(f"{','.join(f[(s, c)]) or '-':14s}" for c in conds))
