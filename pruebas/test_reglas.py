"""Pruebas de las reglas y del gestor de alertas. Ejecutar: python -m unittest discover pruebas"""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.alertas import GestorAlertas  # noqa: E402
from vigia.historial import Historial, Muestra  # noqa: E402
from vigia.reglas import Reglas, angulo_tronco, punto_en_poligono  # noqa: E402
from vigia.simulador import ALTO, ANCHO, ZONAS_DEMO, Simulador, esqueleto  # noqa: E402
from vigia.tipos import Evento, Observacion, Persona  # noqa: E402

CFG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


def reglas_demo():
    r = copy.deepcopy(CFG["reglas"])
    r["merodeo"]["tiempo_max"] = 12
    r["objetos"]["tiempo_abandono"] = 10
    return Reglas(r)


def recorrer(desde, hasta, paso=0.1, zonas=ZONAS_DEMO):
    sim, hist, reglas = Simulador(), Historial(), reglas_demo()
    eventos, t = {}, desde
    while t < hasta:
        _, obs = sim.frame(t)
        hist.actualizar(obs)
        for e in reglas.evaluar(obs, hist, ANCHO, ALTO, zonas):
            eventos.setdefault(e.clave, (t, e))
        t += paso
    return eventos


class Geometria(unittest.TestCase):
    def test_angulo_de_pie_y_tendido(self):
        de_pie = esqueleto(400, 480, "quieto", 0, 0, 1, 0)
        tendido = esqueleto(400, 480, "tendido", 0, 0, 1, 1)
        m = lambda k: Muestra(0, np.array([0, 0, 1, 1.0]), k, np.full(17, 0.9))
        self.assertLess(angulo_tronco(m(de_pie)), 10)
        self.assertGreater(angulo_tronco(m(tendido)), 75)

    def test_punto_en_poligono(self):
        cuadro = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
        self.assertTrue(punto_en_poligono(0.3, 0.3, cuadro))
        self.assertFalse(punto_en_poligono(0.7, 0.3, cuadro))


class Escenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eventos = recorrer(0, 100)

    def test_cada_comportamiento_una_vez(self):
        tipos = sorted(e.tipo for _, e in self.eventos.values())
        self.assertEqual(tipos, sorted(["merodeo", "intrusion", "violencia", "caida", "en_suelo", "abandono", "hurto"]))

    def test_momentos(self):
        cuando = {e.tipo: t for t, e in self.eventos.values()}
        self.assertTrue(26 <= cuando["merodeo"] <= 30)      # entra a la zona ~15 s, límite 12 s
        self.assertTrue(44 <= cuando["violencia"] <= 50)    # golpes entre 45 y 51 s
        self.assertTrue(61 <= cuando["caida"] <= 63)
        self.assertTrue(66 <= cuando["en_suelo"] <= 68)
        self.assertTrue(89 <= cuando["hurto"] <= 92)

    def test_caminar_y_cruzarse_no_es_pelea(self):
        ev = recorrer(0, 14)
        self.assertFalse([e for _, e in ev.values() if e.tipo == "violencia"])

    def test_cada_alerta_trae_explicacion(self):
        for _, e in self.eventos.values():
            self.assertTrue(e.explicacion and any(ch.isdigit() for ch in e.explicacion + e.mensaje))


class SinZonas(unittest.TestCase):
    def test_merodeo_en_toda_la_camara(self):
        r = copy.deepcopy(CFG["reglas"])
        r["merodeo"]["tiempo_max"] = 5
        r["merodeo"]["sin_zonas"] = True          # opcional: apagado por defecto
        reglas, hist = Reglas(r), Historial()
        k = esqueleto(300, 480, "quieto", 0, 0, 1, 0)
        caja = np.array([260, 240, 340, 480.0])
        tipos = set()
        for i in range(70):
            obs = Observacion(t=i * 0.1, personas=[Persona(1, caja, k, np.full(17, 0.9))])
            hist.actualizar(obs)
            tipos |= {e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])}
        self.assertEqual(tipos, {"merodeo"})

    def test_tendido_desde_el_inicio_no_es_caida(self):
        reglas, hist = reglas_demo(), Historial()
        k = esqueleto(400, 480, "tendido", 0, 0, 1, 1)
        caja = np.array([380, 440, 640, 482.0])
        tipos = []
        for i in range(70):
            obs = Observacion(t=i * 0.1, personas=[Persona(1, caja, k, np.full(17, 0.9))])
            hist.actualizar(obs)
            tipos += [e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])]
        self.assertIn("en_suelo", tipos)
        self.assertNotIn("caida", tipos)


def pelea_a_distancia(sep_alturas, segundos=3.0):
    """Dos personas que lanzan golpes, separadas sep_alturas (la altura es H del simulador)."""
    from vigia.simulador import H
    reglas, hist = Reglas(copy.deepcopy(CFG["reglas"])), Historial()
    tipos, t = set(), 0.0
    cx = 400.0
    while t < segundos:
        ps = []
        for pid, x, d in ((1, cx, 1), (2, cx + sep_alturas * H, -1)):
            k = esqueleto(x, 480, "golpea", t, 0, d, 0)
            caja = np.array([k[:, 0].min() - 10, k[:, 1].min() - 10, k[:, 0].max() + 10, 482.0])
            ps.append(Persona(pid, caja, k, np.full(17, 0.9)))
        obs = Observacion(t=t, personas=ps)
        hist.actualizar(obs)
        tipos |= {e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])}
        t += 0.1
    return tipos


class ViolenciaConContacto(unittest.TestCase):
    """Corrección tras la prueba con un video real de intento de robo: mover los brazos rápido cerca de
    otra persona (por ejemplo, sobre una moto) no es pelea si no hay contacto."""

    def test_golpes_que_alcanzan_son_violencia(self):
        self.assertIn("violencia", pelea_a_distancia(0.6))

    def test_brazos_rapidos_sin_alcanzar_no_es_violencia(self):
        self.assertNotIn("violencia", pelea_a_distancia(1.0))


class RoboVehiculo(unittest.TestCase):
    MOTO = np.array([300, 380, 520, 480.0])

    def correr(self, dueno_estaciona, se_mueve=False):
        from vigia.tipos import Objeto
        reglas, ev = Reglas(copy.deepcopy(CFG["reglas"])), []
        k = esqueleto(410, 470, "quieto", 0, 0, 1, 0)
        caja_p = np.array([370, 240, 450, 472.0])
        for i in range(140):                       # 14 s a 10 fps
            t = i * 0.1
            moto = self.MOTO + ([0, 0, 0, 0] if not (se_mueve and t > 11) else [(t - 11) * 60, 0, (t - 11) * 60, 0])
            personas = []
            if dueno_estaciona and t < 1:
                personas.append(Persona(1, caja_p, k, np.full(17, 0.9)))      # llega, la estaciona y se va
            if 7 <= t:
                personas.append(Persona(2, caja_p + (moto - self.MOTO)[[0, 1, 0, 1]], k, np.full(17, 0.9)))
            obs = Observacion(t=t, personas=personas, objetos=[Objeto(9 if t < 8 else 10, "motocicleta", moto)])
            ev += [e for e in reglas.evaluar(obs, Historial(), ANCHO, ALTO, []) if e.tipo == "robo_vehiculo"]
        return ev

    def test_desconocido_sobre_moto_estacionada(self):
        ev = self.correr(dueno_estaciona=False)
        self.assertTrue(ev and ev[0].severidad == "media")
        self.assertIn("no se vio a nadie estacionarlo", ev[0].explicacion)

    def test_otro_distinto_del_dueno_y_se_la_lleva(self):
        ev = self.correr(dueno_estaciona=True, se_mueve=True)
        self.assertIn("persona #1", ev[0].explicacion)
        self.assertTrue(any(e.severidad == "alta" for e in ev))

    def test_el_dueno_no_genera_alerta(self):
        from vigia.tipos import Objeto
        reglas, ev = Reglas(copy.deepcopy(CFG["reglas"])), []
        k = esqueleto(410, 470, "quieto", 0, 0, 1, 0)
        caja_p = np.array([370, 240, 450, 472.0])
        for i in range(140):
            t = i * 0.1
            personas = [Persona(1, caja_p, k, np.full(17, 0.9))] if (t < 1 or t >= 7) else []
            obs = Observacion(t=t, personas=personas, objetos=[Objeto(9, "motocicleta", self.MOTO)])
            ev += [e for e in reglas.evaluar(obs, Historial(), ANCHO, ALTO, []) if e.tipo == "robo_vehiculo"]
        self.assertEqual(ev, [])

    def test_moto_estacionada_no_es_objeto_abandonado(self):
        from vigia.tipos import Objeto
        reglas, hist, tipos = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), set()
        for i in range(400):
            obs = Observacion(t=i * 0.1, objetos=[Objeto(9, "motocicleta", self.MOTO)])
            hist.actualizar(obs)
            tipos |= {e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])}
        self.assertNotIn("abandono", tipos)


def persona(pid, k):
    caja = np.array([k[:, 0].min() - 10, k[:, 1].min() - 10, k[:, 0].max() + 10, k[:, 1].max() + 2])
    return Persona(pid, caja, k, np.full(17, 0.9))


def recorrer_escena(cuadros, segundos, zonas=(), reloj=None, paso=0.1):
    """cuadros(t) -> (personas, objetos). Devuelve los eventos generados con todas las reglas."""
    reglas, hist, ev, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), [], 0.0
    if reloj:
        reglas.reloj = reloj
    while t < segundos:
        ps, objs = cuadros(t)
        obs = Observacion(t=t, personas=ps, objetos=objs)
        hist.actualizar(obs)
        ev += reglas.evaluar(obs, hist, ANCHO, ALTO, list(zonas))
        t += paso
    return ev


class RoboConHuida(unittest.TestCase):
    """Arrebato o asalto: contacto con la víctima y huida corriendo (mapa de actos, prioridad alta)."""

    def escena(self, vel_huida):
        from vigia.simulador import H
        def cuadros(t):
            if t < 1.5:                                    # el agresor alcanza a la víctima
                a = esqueleto(400, 480, "golpea", t, 0, 1, 0)
            else:                                          # y se va hacia la izquierda
                x = 400 - vel_huida * H * (t - 1.5)
                a = esqueleto(x, 480, "camina", t, abs(x - 400), -1, 0)
            v = esqueleto(400 + 0.6 * H, 480, "quieto", t, 0, -1, 0)
            return [persona(1, a), persona(2, v)], []
        return [e for e in recorrer_escena(cuadros, 4.0) if e.tipo == "robo_huida"]

    def test_toca_y_huye_corriendo(self):
        ev = self.escena(vel_huida=2.5)
        self.assertTrue(ev)
        self.assertIn("#1 tocó a #2", ev[0].mensaje)

    def test_toca_y_se_va_caminando_no_es_huida(self):
        self.assertEqual(self.escena(vel_huida=0.7), [])


class Vandalismo(unittest.TestCase):
    def escena(self, con_auto=True, acompanado=False, zonas=()):
        from vigia.simulador import H
        from vigia.tipos import Objeto
        auto = np.array([400 + 0.35 * H, 480 - 1.0 * H, 400 + 1.6 * H, 480.0])
        def cuadros(t):
            ps = [persona(1, esqueleto(400, 480, "golpea", t, 0, 1, 0))]
            if acompanado:
                ps.append(persona(2, esqueleto(400 - 0.8 * H, 480, "quieto", t, 0, 1, 0)))
            return ps, ([Objeto(9, "auto", auto)] if con_auto else [])
        return {e.tipo for e in recorrer_escena(cuadros, 3.0, zonas)}

    def test_golpes_contra_un_auto(self):
        tipos = self.escena()
        self.assertIn("vandalismo", tipos)
        self.assertNotIn("violencia", tipos)

    def test_sin_vehiculo_ni_zona_no_hay_vandalismo(self):
        self.assertNotIn("vandalismo", self.escena(con_auto=False))

    def test_con_otra_persona_cerca_no_es_vandalismo(self):
        self.assertNotIn("vandalismo", self.escena(acompanado=True))

    def test_zona_protegida(self):
        vitrina = {"id": "v1", "nombre": "Vitrina", "tipo": "protegida",
                   "puntos": [[0.48, 0.30], [0.80, 0.30], [0.80, 0.95], [0.48, 0.95]]}
        self.assertIn("vandalismo", self.escena(con_auto=False, zonas=[vitrina]))


class Armas(unittest.TestCase):
    def escena(self, en_la_mano):
        from vigia.tipos import Objeto
        k = esqueleto(400, 480, "quieto", 0, 0, 1, 0)
        donde = k[9] if en_la_mano else np.array([800.0, 470.0])
        cuchillo = np.array([donde[0] - 8, donde[1] - 20, donde[0] + 8, donde[1] + 5])
        cuadros = lambda t: ([persona(1, k)], [Objeto(5, "cuchillo", cuchillo)])
        return [e for e in recorrer_escena(cuadros, 1.0) if e.tipo == "arma"]

    def test_cuchillo_en_la_mano(self):
        ev = self.escena(en_la_mano=True)
        self.assertTrue(ev and "cuchillo" in ev[0].mensaje)

    def test_cuchillo_lejos_de_las_manos(self):
        self.assertEqual(self.escena(en_la_mano=False), [])

    def test_un_cuchillo_no_es_objeto_abandonado(self):
        from vigia.tipos import Objeto
        caja = np.array([600, 450, 620, 470.0])
        tipos = {e.tipo for e in recorrer_escena(lambda t: ([], [Objeto(5, "cuchillo", caja)]), 40.0)}
        self.assertNotIn("abandono", tipos)


class ZonasConHorario(unittest.TestCase):
    ZONA = {"id": "z1", "nombre": "Almacén", "tipo": "restringida", "horario": {"desde": "20:00", "hasta": "23:00"},
            "puntos": [[0.3, 0.7], [0.6, 0.7], [0.6, 0.95], [0.3, 0.95]]}

    def tipos_a_las(self, hh):
        from datetime import time
        k = esqueleto(400, 480, "quieto", 0, 0, 1, 0)
        ev = recorrer_escena(lambda t: ([persona(1, k)], []), 1.0, [self.ZONA], reloj=lambda: time(hh, 0))
        return {e.tipo for e in ev}

    def test_dentro_del_horario_alerta(self):
        self.assertIn("intrusion", self.tipos_a_las(21))

    def test_fuera_del_horario_no_alerta(self):
        self.assertNotIn("intrusion", self.tipos_a_las(12))

    def test_horario_que_cruza_la_medianoche(self):
        from datetime import time
        from vigia.reglas import zona_activa
        z = {"horario": {"desde": "22:00", "hasta": "06:00"}}
        self.assertTrue(zona_activa(z, time(23, 30)))
        self.assertTrue(zona_activa(z, time(3, 0)))
        self.assertFalse(zona_activa(z, time(12, 0)))
        self.assertTrue(zona_activa({}, time(12, 0)))          # sin horario: siempre


class Aglomeracion(unittest.TestCase):
    def multitud(self, n, vel=0.0):
        from vigia.simulador import H
        def cuadros(t):
            ps = []
            for i in range(n):
                x = 60 + i * 55 + vel * H * t
                ps.append(persona(i + 1, esqueleto(x, 300 + (i % 3) * 90, "camina" if vel else "quieto",
                                                    t, vel * H * t, 1, 0)))
            return ps, []
        return {e.tipo for e in recorrer_escena(cuadros, 6.5)}

    def test_muchas_personas(self):
        self.assertIn("aglomeracion", self.multitud(16))

    def test_pocas_personas(self):
        self.assertNotIn("aglomeracion", self.multitud(10))

    def test_huida_masiva(self):
        tipos = self.multitud(5, vel=2.5)
        self.assertIn("estampida", tipos)
        self.assertNotIn("aglomeracion", tipos)

    def test_caminar_en_grupo_no_es_huida(self):
        self.assertNotIn("estampida", self.multitud(5, vel=0.7))


class SoloEsqueleto(unittest.TestCase):
    """Privacidad por diseño: la imagen de la persona se borra y el resto de la escena se conserva."""

    def test_borra_a_la_persona_y_deja_el_fondo(self):
        from vigia.dibujo import FONDO_PERSONA, solo_esqueleto
        img = np.full((300, 400, 3), 200, np.uint8)
        obs = Observacion(t=0, personas=[Persona(1, np.array([100, 50, 160, 250.0]), None, None)])
        solo_esqueleto(img, obs)
        self.assertTrue((img[150, 130] == FONDO_PERSONA).all())     # dentro de la persona
        self.assertTrue((img[150, 350] == 200).all())               # fondo intacto

    def test_el_aviso_lo_declara(self):
        from vigia.aviso import pagina_aviso
        cfg = copy.deepcopy(CFG)
        cfg["privacidad"]["solo_esqueleto"] = True
        self.assertIn("solo una figura de su esqueleto", pagina_aviso(cfg, 1))


class Encuesta(unittest.TestCase):
    def setUp(self):
        from vigia.encuesta import Encuesta as E
        self.dir = tempfile.TemporaryDirectory()
        self.e = E(Path(self.dir.name))

    def tearDown(self):
        self.e.cerrar()
        self.dir.cleanup()

    def test_solo_mayores_de_edad(self):
        with self.assertRaises(ValueError):
            self.e.guardar({"general": 5})
        with self.assertRaises(ValueError):
            self.e.guardar({"mayor_de_edad": False, "general": 5})
        self.assertEqual(self.e.resultados()["n"], 0)

    def test_porcentajes_y_respuestas_invalidas(self):
        self.e.guardar({"mayor_de_edad": True, "general": 5, "facial": 1, "confianza": ["Universidad", "Hackers"],
                        "rol": "Docente o personal"})
        self.e.guardar({"mayor_de_edad": True, "general": 4, "facial": 9, "confianza": ["Universidad", "Municipalidad"]})
        self.e.guardar({"mayor_de_edad": True, "general": 2})
        r = {p["id"]: p for p in self.e.resultados()["preguntas"]}
        self.assertEqual(r["general"]["acuerdo"], 66.7)                    # 2 de 3 respondieron 4 o 5
        self.assertEqual(r["general"]["ref"]["valor"], 82.2)                # referencia E45 del artículo
        self.assertEqual(r["facial"]["n"], 1)                               # el 9 se descarta
        self.assertEqual(r["confianza"]["conteo"]["Universidad"], 2)
        self.assertNotIn("Hackers", r["confianza"]["conteo"])
        self.assertEqual(r["confianza"]["porcentaje"]["Universidad"], 100.0)
        csv_txt = self.e.exportar_csv().splitlines()
        self.assertEqual(len(csv_txt), 4)
        self.assertIn("Municipalidad; Universidad", csv_txt[2])          # en el orden de las opciones


class Retos(unittest.TestCase):
    def test_marcador(self):
        from vigia.reto import Marcador
        with tempfile.TemporaryDirectory() as d:
            m = Marcador(Path(d) / "reto.json")
            m.registrar("caida", True, 4.2)
            m.registrar("caida", True, 6.0)
            r = m.registrar("caida", False, None)["caida"]
            self.assertEqual((r["intentos"], r["detectados"], r["tasa"]), (3, 2, 66.7))
            self.assertEqual(Marcador(Path(d) / "reto.json").resumen()["caida"]["intentos"], 3)   # persiste
            with self.assertRaises(ValueError):
                m.registrar("otro", True, 1)

    def test_cada_reto_corresponde_a_una_regla(self):
        from vigia.alertas import NOMBRES
        from vigia.reto import RETOS
        for r in RETOS:
            self.assertTrue(all(t in NOMBRES for t in r.get("tipos", [])) or r.get("emocion"))


class FakeCLIP:
    """Codificador de prueba: el vector de una imagen es su color medio (B, G, R); «rojo» apunta al rojo."""
    COLORES = {"rojo": [0, 0, 1.0], "verde": [0, 1.0, 0], "azul": [1.0, 0, 0]}

    def imagen(self, ruta):
        import cv2
        v = cv2.imread(str(ruta)).reshape(-1, 3).mean(0)
        return v / np.linalg.norm(v)

    def texto(self, frase):
        v = np.array(self.COLORES[frase], float)
        return v / np.linalg.norm(v)


class Busqueda(unittest.TestCase):
    def esperar(self, b, total):
        import time
        for _ in range(100):
            if b.estado == "lista" and b.total >= total and b.pendientes() == 0:
                return
            time.sleep(0.05)

    def test_indexa_busca_y_persiste(self):
        import cv2
        from vigia.busqueda import Buscador
        with tempfile.TemporaryDirectory() as d:
            cap = Path(d) / "capturas"
            cap.mkdir()
            for nombre, bgr in (("a_rojo.jpg", (20, 20, 230)), ("b_verde.jpg", (20, 230, 20))):
                cv2.imwrite(str(cap / nombre), np.full((40, 60, 3), bgr, np.uint8))
            b = Buscador(cap, Path(d) / "indice.npz", fabrica=FakeCLIP)
            with self.assertRaises(RuntimeError):
                b.buscar("rojo")                                   # aún apagada
            b.iniciar("x", "y")
            self.esperar(b, 2)                                     # indexa las capturas que ya existían
            cv2.imwrite(str(cap / "c_azul.jpg"), np.full((40, 60, 3), (230, 20, 20), np.uint8))
            b.agregar("c_azul.jpg")                                # y las nuevas, al guardarse
            self.esperar(b, 3)
            self.assertEqual(b.buscar("rojo", 1)[0]["captura"], "a_rojo.jpg")
            self.assertEqual(b.buscar("azul", 1)[0]["captura"], "c_azul.jpg")
            (cap / "a_rojo.jpg").unlink()                          # borrada por retención: no aparece
            self.assertNotIn("a_rojo.jpg", [r["captura"] for r in b.buscar("rojo", 5)])
            b.detener()
            self.assertEqual(Buscador(cap, Path(d) / "indice.npz").total, 3)   # el índice se guarda en disco

    def test_sin_open_clip_avisa_que_instalar(self):
        from vigia.busqueda import Buscador

        def falta():
            raise ImportError("open_clip")
        with tempfile.TemporaryDirectory() as d:
            b = Buscador(Path(d), Path(d) / "i.npz", fabrica=falta)
            b.iniciar("x", "y")
            b._hilo.join(2)
            self.assertIn("requirements-busqueda.txt", b.estado)


class ModoFeria(unittest.TestCase):
    """Desde otros equipos de la red solo se abre la encuesta; el panel y las alertas siguen siendo locales."""

    def pedir(self, cliente, metodo, ruta):
        import asyncio
        from vigia.servidor import SoloLocal
        enviados = []

        async def interno(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def recibir():
            return {"type": "http.request", "body": b""}

        async def enviar(m):
            enviados.append(m)
        scope = {"type": "http", "method": metodo, "path": ruta, "client": (cliente, 5000), "headers": [],
                 "query_string": b""}
        asyncio.run(SoloLocal(interno)(scope, recibir, enviar))
        return enviados[0]["status"]

    def test_permisos(self):
        self.assertEqual(self.pedir("192.168.1.20", "GET", "/encuesta"), 200)
        self.assertEqual(self.pedir("192.168.1.20", "POST", "/api/encuesta"), 200)
        self.assertEqual(self.pedir("192.168.1.20", "GET", "/static/encuesta.js"), 200)
        self.assertEqual(self.pedir("192.168.1.20", "GET", "/"), 403)
        self.assertEqual(self.pedir("192.168.1.20", "GET", "/api/alertas"), 403)
        self.assertEqual(self.pedir("192.168.1.20", "PUT", "/api/ajustes"), 403)
        self.assertEqual(self.pedir("192.168.1.20", "GET", "/api/encuesta.csv"), 403)
        self.assertEqual(self.pedir("127.0.0.1", "GET", "/api/alertas"), 200)

    def test_tiempos_cortos(self):
        from vigia.servidor import Sistema
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(CFG)
            cfg["camaras"] = []
            s = Sistema(cfg, Path(d), demo=False, feria=True)
            self.assertEqual(s.cfg["reglas"]["merodeo"]["tiempo_max"], 10)
            self.assertEqual(s.alertas.enfriamiento, 5)
            self.assertEqual(CFG["reglas"]["merodeo"]["tiempo_max"], 60)     # el config original no cambia
            s.alertas.cerrar()
            s.encuesta.cerrar()


class EnlacesDeVideo(unittest.TestCase):
    """Un enlace de YouTube se descarga con yt-dlp (aquí, uno simulado) y se agrega como cámara."""

    def yt_dlp_falso(self, origen: Path, duracion: int):
        import shutil
        import types

        class YoutubeDL:
            def __init__(self, op):
                self.op = op

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                return {"title": "Pelea en la calle", "duration": duracion, "ext": "mp4"}

            def process_ie_result(self, info, download=True):
                destino = Path(self.op["outtmpl"].replace("%(ext)s", "mp4"))
                shutil.copy(origen, destino)
                for h in self.op["progress_hooks"]:
                    h({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                return {**info, "requested_downloads": [{"filepath": str(destino)}]}
        return types.SimpleNamespace(YoutubeDL=YoutubeDL)

    def descargar(self, duracion=60, video=True):
        import sys
        import time
        import cv2
        from vigia.descargas import Descargas
        d = tempfile.mkdtemp()
        origen = Path(d) / "origen.mp4"
        if video:
            w = cv2.VideoWriter(str(origen), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for _ in range(5):
                w.write(np.zeros((48, 64, 3), np.uint8))
            w.release()
        else:
            origen.write_bytes(b"no es un video")
        sys.modules["yt_dlp"] = self.yt_dlp_falso(origen, duracion)
        try:
            agregadas = []
            des = Descargas(Path(d) / "videos")
            tid = des.iniciar("https://youtu.be/abc", "", lambda ruta, t: agregadas.append((ruta, t)) or {"id": "x"})
            for _ in range(100):
                if des.estado(tid)["estado"] in ("lista", "error"):
                    break
                time.sleep(0.05)
            return des.estado(tid), agregadas, Path(d) / "videos"
        finally:
            del sys.modules["yt_dlp"]

    def test_descarga_y_agrega_la_camara(self):
        est, agregadas, carpeta = self.descargar()
        self.assertEqual(est["estado"], "lista")
        self.assertEqual(agregadas[0][1], "Pelea en la calle")             # título del video si no se dio nombre
        self.assertEqual(Path(agregadas[0][0]).parent, carpeta)             # queda en datos/videos (se borra al quitarla)

    def test_video_demasiado_largo(self):
        est, agregadas, _ = self.descargar(duracion=3600)
        self.assertEqual(est["estado"], "error")
        self.assertIn("límite", est["error"])
        self.assertEqual(agregadas, [])

    def test_formato_ilegible(self):
        est, agregadas, carpeta = self.descargar(video=False)
        self.assertEqual(est["estado"], "error")
        self.assertEqual(list(carpeta.glob("*")), [])                      # no deja archivos sueltos


class Escenario_aprendido(unittest.TestCase):
    """Falsas alarmas vistas en el video de una tienda: prendas y bolsas del escaparate se marcaban como
    abandonadas porque el primer cliente que pasaba cerca quedaba como su «dueño»."""

    def correr(self, objetos_en, personas_en, segundos=50.0, ignorar=None, sin_zonas=False):
        from vigia.tipos import Objeto
        r = copy.deepcopy(CFG["reglas"])
        r["merodeo"]["sin_zonas"] = sin_zonas
        reglas, hist, tipos, t = Reglas(r), Historial(), [], 0.0
        if ignorar:
            reglas.aprender_ignorar(ignorar)
        k = esqueleto(560, 470, "quieto", 0, 0, 1, 0)
        bolsa = np.array([600, 430, 640, 470.0])
        while t < segundos:
            ps = [persona(1, k)] if personas_en(t) else []
            obs = Observacion(t=t, personas=ps, objetos=[Objeto(4, "bolso", bolsa)] if objetos_en(t) else [])
            hist.actualizar(obs)
            tipos += [e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])]
            t += 0.1
        return tipos

    def test_lo_que_estaba_desde_el_inicio_no_se_abandona(self):
        # el bolso está desde el primer cuadro; un cliente pasa a su lado entre los 15 y 18 s
        self.assertNotIn("abandono", self.correr(lambda t: True, lambda t: 15 <= t < 18))

    def test_lo_que_aparece_sin_nadie_no_se_abandona(self):
        self.assertNotIn("abandono", self.correr(lambda t: t >= 12, lambda t: 20 <= t < 22))

    def test_lo_que_alguien_trae_y_deja_si(self):
        # llega una persona a los 12 s, aparece el bolso a su lado, ella se va a los 16 s
        self.assertIn("abandono", self.correr(lambda t: t >= 13, lambda t: 12 <= t < 16))

    def test_lo_que_el_operador_marco_como_falso_no_vuelve(self):
        tipos = self.correr(lambda t: t >= 13, lambda t: 12 <= t < 16, ignorar=[600, 430, 640, 470])
        self.assertNotIn("abandono", tipos)

    def test_merodeo_sin_zonas_apagado_por_defecto(self):
        self.assertNotIn("merodeo", self.correr(lambda t: False, lambda t: True, segundos=70))
        self.assertIn("merodeo", self.correr(lambda t: False, lambda t: True, segundos=70, sin_zonas=True))


class AgruparYAprender(unittest.TestCase):
    def test_alertas_repetidas_se_suman(self):
        with tempfile.TemporaryDirectory() as d:
            g = GestorAlertas(Path(d), enfriamiento=0, agrupar=60)
            frame = np.zeros((60, 80, 3), np.uint8)
            for i in range(5):
                g.procesar("cam1", [Evento("merodeo", "media", f"m{i}", "x 1", f"merodeo:{i}", [[1, 1, 9, 9]])], frame)
            g.procesar("cam1", [Evento("caida", "alta", "c", "x 1", "caida:1")], frame)       # otro tipo: otra fila
            filas = g.listar()
            self.assertEqual(len(filas), 2)
            self.assertEqual({a["tipo"]: a["repeticiones"] for a in filas}, {"merodeo": 5, "caida": 1})
            self.assertEqual(g.resumen()["merodeo"], {"total": 5, "pendientes": 1})
            aid = [a["id"] for a in filas if a["tipo"] == "merodeo"][0]
            g.atender(aid)                                                                  # atendida: lo nuevo es otra alerta
            g.procesar("cam1", [Evento("merodeo", "media", "m9", "x 1", "merodeo:9")], frame)
            self.assertEqual(len(g.listar()), 3)
            g.cerrar()

    def test_falsa_alarma_ensena_al_sistema(self):
        with tempfile.TemporaryDirectory() as d:
            g = GestorAlertas(Path(d), enfriamiento=0)
            aprendido = []
            g.al_descartar = lambda cam, tipo, cajas: aprendido.append((cam, tipo, cajas))
            frame = np.zeros((60, 80, 3), np.uint8)
            a = g.procesar("cam1", [Evento("abandono", "media", "m", "x 1", "abandono:4", [[600, 430, 640, 470]])], frame)[0]
            g.dar_veredicto(a["id"], "falsa")
            self.assertEqual(aprendido, [("cam1", "abandono", [[600.0, 430.0, 640.0, 470.0]])])
            g.cerrar()


class EvaluacionConVideosPropios(unittest.TestCase):
    """scripts/evaluar_videos.py: mismas reglas que una cámara, métricas por regla con intervalo de confianza."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluar_videos", RAIZ / "scripts" / "evaluar_videos.py")
        cls.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ev)

    def test_wilson(self):
        lo, hi = self.ev.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.490, places=2)
        self.assertAlmostEqual(hi, 0.943, places=2)

    def test_resumen_por_regla(self):
        filas = [{"tipo": "caida", "etiqueta": "positivo", "detectado": True, "segundos": 3.0, "n_alertas": 1, "duracion": 10},
                 {"tipo": "caida", "etiqueta": "positivo", "detectado": False, "segundos": None, "n_alertas": 0, "duracion": 10},
                 {"tipo": "caida", "etiqueta": "normal", "detectado": True, "segundos": 2.0, "n_alertas": 1, "duracion": 10},
                 {"tipo": "caida", "etiqueta": "normal", "detectado": False, "segundos": None, "n_alertas": 0, "duracion": 10},
                 {"tipo": None, "etiqueta": "escena", "detectado": False, "segundos": None, "n_alertas": 3, "duracion": 1800}]
        r = self.ev.resumir(filas)
        m = r["por_tipo"]["caida"]
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual((m["sensibilidad"], m["precision"], m["especificidad"]), (0.5, 0.5, 0.5))
        self.assertEqual(r["escenas"]["por_hora"], 6.0)                      # 3 alertas en media hora
        texto = self.ev.ficha(r, filas, Path("x"), CFG, 10, 50.0, "CPU de prueba", {"caida": (3, 1)})
        self.assertIn("| Caída | 2/2 | 50 %", texto)
        self.assertIn("6.0 por hora", texto)
        self.assertIn("| Caída | 3 | 1 | 75 % |", texto)

    def test_analiza_un_video_con_el_mismo_flujo_que_una_camara(self):
        import cv2

        class DetectorDemo:                       # en vez de YOLO, la escena simulada del modo demo
            def __init__(self):
                self.sim = Simulador()

            def procesar(self, frame, t):
                return self.sim.frame(t)[1]
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "escena.mp4"
            w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for _ in range(700):                  # 70 s: incluye la caída del guion (a los 62 s)
                w.write(np.zeros((48, 64, 3), np.uint8))
            w.release()
            r = self.ev.analizar_video(ruta, CFG, DetectorDemo(), fps=10)
        tipos = {tipo for _, tipo in r["eventos"]}
        self.assertIn("caida", tipos)
        self.assertIn("violencia", tipos)
        self.assertAlmostEqual(r["duracion"], 70, delta=0.5)

    def test_estructura_de_carpetas(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            for rel in ("caida/positivos/a.mp4", "caida/normales/b.mp4", "normales/c.mp4", "otra_cosa/positivos/d.mp4"):
                (b / rel).parent.mkdir(parents=True, exist_ok=True)
                (b / rel).write_bytes(b"")
            lista = {(v["tipo"], v["etiqueta"], v["ruta"].name) for v in self.ev.videos_de(b)}
        self.assertEqual(lista, {("caida", "positivo", "a.mp4"), ("caida", "normal", "b.mp4"),
                                 (None, "escena", "c.mp4")})              # «otra_cosa» no es un tipo: se omite


class Alertas(unittest.TestCase):
    def test_enfriamiento_y_persistencia(self):
        with tempfile.TemporaryDirectory() as d:
            g = GestorAlertas(Path(d), enfriamiento=30)
            e = Evento("caida", "alta", "Caída de la persona #1", "prueba 1", "caida:1", [[1, 1, 20, 40]])
            frame = np.zeros((60, 80, 3), np.uint8)
            self.assertEqual(len(g.procesar("cam1", [e], frame)), 1)
            self.assertEqual(len(g.procesar("cam1", [e], frame)), 0)      # repetida: se ignora
            self.assertEqual(len(g.procesar("cam2", [e], frame)), 1)      # otra cámara: sí
            lista = g.listar()
            self.assertEqual(len(lista), 2)
            self.assertTrue((Path(d) / "capturas" / lista[0]["captura"]).exists())
            g.atender(lista[0]["id"])
            self.assertEqual(g.listar()[0]["atendida"], 1)
            g.cerrar()

    def test_veredicto_del_operador(self):
        with tempfile.TemporaryDirectory() as d:
            g = GestorAlertas(Path(d), enfriamiento=0)
            frame = np.zeros((60, 80, 3), np.uint8)
            ids = [g.procesar("cam1", [Evento("caida", "alta", "m", "x 1", f"caida:{i}")], frame)[0]["id"] for i in range(4)]
            g.dar_veredicto(ids[0], "confirmada")
            g.dar_veredicto(ids[1], "confirmada")
            g.dar_veredicto(ids[2], "falsa")
            with self.assertRaises(ValueError):
                g.dar_veredicto(ids[3], "quizas")
            d_op = g.desempeno_operador()
            self.assertEqual(d_op["total"], {"confirmadas": 2, "falsas": 1, "precision": 0.667})
            self.assertEqual(d_op["por_tipo"]["caida"]["falsas"], 1)
            csv_txt = g.exportar_csv()
            self.assertIn("veredicto", csv_txt.splitlines()[0])
            self.assertEqual(len(csv_txt.strip().splitlines()), 5)
            g.cerrar()


class Evaluacion(unittest.TestCase):
    """El script de evaluación reporta AUC con un puntaje continuo (Tabla 11 del artículo)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluar", RAIZ / "scripts" / "evaluar_rwf2000.py")
        cls.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ev)

    def test_auc(self):
        self.assertEqual(self.ev.auc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]), 1.0)
        self.assertEqual(self.ev.auc([1, 0], [0.5, 0.5]), 0.5)

    def test_puntaje_pelea_mayor_que_caminar(self):
        sim = Simulador()
        clip = lambda t0: [sim.frame(t0 + i / 10)[1] for i in range(40)]
        pelea = self.ev.puntaje(clip(45), CFG["reglas"])
        caminar = self.ev.puntaje(clip(2), CFG["reglas"])
        self.assertGreater(pelea, 1.8)
        self.assertLess(caminar, pelea)


class Aviso(unittest.TestCase):
    def test_aviso_declara_finalidad_y_limites(self):
        from vigia.aviso import pagina_aviso
        html = pagina_aviso(CFG, 2)
        self.assertIn(CFG["privacidad"]["finalidad"], html)
        self.assertIn("No identifica a las personas", html)
        self.assertIn("29733", html)
        self.assertNotIn("expresiones faciales", html)      # emociones desactivadas por defecto


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------- posiciones de cámara y hurto real
def _persona(pid, k):
    from vigia.simulador import _caja
    return Persona(pid, _caja(k), k, np.full(17, 0.9))


def _comprimir(k, pies, factor_y, cx=None):
    """Pose vista por una cámara elevada cuando la persona cae hacia ella o alejándose: el cuerpo queda casi
    vertical en la imagen pero aplastado contra el piso (no se ve el tronco horizontal)."""
    k = k.copy()
    k[:, 1] = pies - (pies - k[:, 1]) * factor_y
    if cx is not None:
        k[:, 0] = cx + (k[:, 0] - cx) * 0.6
    return k


class PosicionesDeCamara(unittest.TestCase):
    def correr(self, poses):
        """poses(t) -> puntos de la persona 1; devuelve los tipos de alerta."""
        reglas, hist, tipos, t = reglas_demo(), Historial(), [], 0.0
        while t < 12:
            obs = Observacion(t=t, personas=[_persona(1, poses(t))])
            hist.actualizar(obs)
            tipos += [e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])]
            t += 0.1
        return tipos

    def test_inclinarse_a_recoger_algo_no_es_caida(self):
        from vigia.simulador import H
        de_pie = esqueleto(400, 480, "quieto", 0, 0, 1, 0)

        def poses(t):
            if t < 4 or t > 7:
                return de_pie
            # se agacha a fondo para recoger algo, como en la prueba real (la cadera bajó 0.38 alturas):
            # tronco horizontal hacia adelante, cadera cerca del piso y canillas verticales
            k = de_pie.copy()
            cad = np.array([400.0, 480 - 0.10 * H])
            k[[11, 12]] = cad + np.array([[0.05 * H, 0], [-0.05 * H, 0]])
            k[[13, 14]] = np.array([[430.0, 480 - 0.22 * H], [420.0, 480 - 0.22 * H]])
            k[[15, 16]] = np.array([[430.0, 480.0], [420.0, 480.0]])
            k[[5, 6]] = cad + np.array([[0.30 * H, -0.06 * H], [0.30 * H, -0.02 * H]])
            k[[7, 8]] = cad + np.array([[0.34 * H, 0.05 * H], [0.32 * H, 0.06 * H]])
            k[[9, 10]] = cad + np.array([[0.36 * H, 0.09 * H], [0.34 * H, 0.09 * H]])
            k[0:5] = cad + np.array([0.45 * H, 0.0])
            return k
        tipos = self.correr(poses)
        self.assertNotIn("caida", tipos)
        self.assertNotIn("en_suelo", tipos)

    def test_caida_hacia_una_camara_elevada(self):
        de_pie = esqueleto(400, 480, "quieto", 0, 0, 1, 0)

        def poses(t):
            if t < 4:
                return de_pie
            a = min(1.0, (t - 4) / 0.8)              # cae en 0.8 s y se queda en el suelo
            return _comprimir(de_pie, 480, 1 - 0.8 * a, 400)
        tipos = self.correr(poses)
        self.assertIn("caida", tipos)
        self.assertIn("en_suelo", tipos)

    def test_caida_de_costado_frente_a_la_camara(self):
        def poses(t):
            if t < 4:
                return esqueleto(400, 480, "quieto", 0, 0, 1, 0)
            return esqueleto(400, 480, "cae" if t < 4.7 else "tendido", 0, 0, 1, min(1.0, (t - 4) / 0.7))
        self.assertIn("caida", self.correr(poses))

    def test_sentarse_en_el_piso_no_es_tendido(self):
        from vigia.simulador import H
        de_pie = esqueleto(400, 480, "quieto", 0, 0, 1, 0)

        def poses(t):
            if t < 4:
                return de_pie
            k = de_pie.copy()                        # tronco erguido, cadera y pies en el piso
            k[:11, 1] += 0.45 * H
            k[[11, 12], 1] = 470
            k[[13, 14]] = k[[11, 12]] + (0.25 * H, -0.05 * H)
            k[[15, 16]] = k[[11, 12]] + (0.45 * H, 10)
            return k
        self.assertNotIn("en_suelo", self.correr(poses))


class HurtoReal(unittest.TestCase):
    """Con un detector real, al levantar la mochila la mano la tapa y el seguimiento le cambia el número
    o la pierde. El hurto tiene que detectarse igual."""

    def escena(self, final):
        from vigia.tipos import Objeto
        reglas, hist, eventos, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), [], 0.0
        bolsa = np.array([280, 440, 330, 480.0])
        while t < 26:
            personas, objetos = [], []
            if t < 12:
                personas.append(_persona(1, esqueleto(300, 480, "quieto", t, 0, 1, 0)))
            elif t < 16:                              # deja la mochila y se va
                x = 300 + (t - 12) * 100
                personas.append(_persona(1, esqueleto(x, 480, "camina", t, x, 1, 0)))
                objetos.append(Objeto(50, "mochila", bolsa))
            elif t < 19.0:                            # la mochila sola; se acerca otra persona
                x = 800 - (t - 16) * 160
                personas.append(_persona(2, esqueleto(x, 480, "camina", t, x, -1, 0)))
                objetos.append(Objeto(50, "mochila", bolsa))
            else:
                personas, objetos = final(t)
            obs = Observacion(t=t, personas=personas, objetos=objetos)
            hist.actualizar(obs)
            eventos += reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        return [e for e in eventos if e.tipo == "hurto"]

    def test_la_mochila_cambia_de_numero_al_levantarla(self):
        from vigia.tipos import Objeto

        def final(t):
            x = 320 - max(0.0, t - 19.6) * 120
            per = [_persona(2, esqueleto(x, 480, "camina", t, x, -1, 0))]
            obj = [] if t < 19.6 else [Objeto(51, "mochila", np.array([x - 25, 360, x + 25, 400.0]))]
            return per, obj
        h = self.escena(final)
        self.assertTrue(h and h[0].severidad == "alta" and "#2" in h[0].mensaje)

    def test_la_mochila_desaparece_con_quien_se_la_lleva(self):
        def final(t):
            x = 320 + max(0.0, t - 19.5) * 200          # se va sin que el detector vuelva a ver la mochila
            return ([_persona(2, esqueleto(x, 480, "camina", t, x, 1, 0))] if x < 960 else []), []
        h = self.escena(final)
        self.assertTrue(h and "#2" in h[0].mensaje)

    def test_el_dueno_recupera_su_mochila(self):
        from vigia.tipos import Objeto

        def final(t):
            x = 320 - max(0.0, t - 19.6) * 120
            per = [_persona(1, esqueleto(x, 480, "camina", t, x, -1, 0))]
            obj = [] if t < 19.6 else [Objeto(51, "mochila", np.array([x - 25, 360, x + 25, 400.0]))]
            return per, obj
        self.assertEqual(self.escena(final), [])

    def test_alguien_tapa_la_mochila_sin_llevarsela(self):
        from vigia.tipos import Objeto

        def final(t):
            per = [_persona(2, esqueleto(320, 480, "quieto", t, 0, -1, 0))]    # se queda parado delante
            obj = [Objeto(52, "mochila", np.array([280, 440, 330, 480.0]))] if t > 23 else []
            return per, obj
        self.assertEqual(self.escena(final), [])

    def test_dueno_aunque_su_cuerpo_tape_la_mochila_al_dejarla(self):
        from vigia.tipos import Objeto
        reglas, hist, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), 0.0
        while t < 16:
            x = 300 if t < 12 else 300 + (t - 12) * 150   # se aleja antes de que el detector vea la mochila
            per = [_persona(1, esqueleto(x, 480, "quieto" if t < 12 else "camina", t, x, 1, 0))]
            obj = [Objeto(60, "mochila", np.array([280, 440, 330, 480.0]))] if t >= 13 else []
            obs = Observacion(t=t, personas=per, objetos=obj)
            hist.actualizar(obs)
            reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        self.assertEqual(hist.objetos[60].estado.get("dueno"), 1)


def _recortar(p, modo):
    """La cámara solo ve medio cuerpo: «piernas» (de la cintura para abajo) o «torso» (de la cintura para arriba)."""
    from vigia.simulador import H
    k, conf = p.puntos.copy(), np.full(17, 0.9)
    vis = list(range(11, 17)) if modo == "piernas" else list(range(0, 13))
    conf[[i for i in range(17) if i not in vis]] = 0.0
    ys = k[vis, 1]
    caja = p.caja.copy()
    if modo == "piernas":
        caja[1] = ys.min() - 0.05 * H
    else:
        caja[3] = ys.max() + 0.05 * H
    return Persona(p.id, caja, k, conf)


class MedioCuerpo(unittest.TestCase):
    """Cámaras que solo ven a las personas de la cintura para arriba (mostrador) o para abajo (piso)."""

    def escena(self, modo, bolsa, ladron):
        from vigia.tipos import Objeto
        from vigia.simulador import H
        reglas, hist, eventos, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), [], 0.0
        alto_bolsa = bolsa[3] - bolsa[1]
        xb = (bolsa[0] + bolsa[2]) / 2 + 0.55 * H      # se la deja y se la toma con el brazo estirado
        while t < 26:
            per, obj = [], []
            if t < 12:
                per.append(_persona(1, esqueleto(xb, 480, "quieto", t, 0, 1, 0)))
            elif t < 16:
                x = xb + (t - 12) * 100
                per.append(_persona(1, esqueleto(x, 480, "camina", t, x, 1, 0)))
                obj.append(Objeto(50, "mochila", bolsa))
            elif t < 19.0:
                x = 900 - (t - 16) * (900 - xb) / 3
                per.append(_persona(2, esqueleto(x, 480, "camina", t, x, -1, 0)))
                obj.append(Objeto(50, "mochila", bolsa))
            else:
                x = xb - max(0.0, t - 19.6) * 120
                per.append(_persona(ladron, esqueleto(x, 480, "camina", t, x, -1, 0)))
                if t >= 19.6:
                    y = bolsa[1] - 60
                    obj.append(Objeto(51, "mochila", np.array([x - 0.45 * H - 25, y, x - 0.45 * H + 25, y + alto_bolsa])))
            obs = Observacion(t=t, personas=[_recortar(p, modo) for p in per], objetos=obj)
            hist.actualizar(obs)
            eventos += reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        return [e for e in eventos if e.tipo == "hurto"]

    def test_altura_estimada_con_medio_cuerpo(self):
        from vigia.simulador import H
        p = _persona(1, esqueleto(300, 480, "quieto", 0, 0, 1, 0))
        for modo in ("piernas", "torso"):
            m = _recortar(p, modo)
            self.assertLess(m.caja[3] - m.caja[1], 0.7 * H)     # la caja mide medio cuerpo…
            self.assertGreater(m.alto, 0.85 * H)                # …pero la estatura se recupera

    def test_hurto_viendo_solo_las_piernas(self):
        h = self.escena("piernas", np.array([280, 440, 330, 480.0]), ladron=2)
        self.assertTrue(h and "#2" in h[0].mensaje)

    def test_hurto_en_un_mostrador_viendo_solo_el_torso(self):
        h = self.escena("torso", np.array([280, 330, 330, 370.0]), ladron=2)
        self.assertTrue(h and "#2" in h[0].mensaje)

    def test_el_dueno_recupera_su_mochila_viendo_medio_cuerpo(self):
        self.assertEqual(self.escena("piernas", np.array([280, 440, 330, 480.0]), ladron=1), [])


class FiltroDeObjetos(unittest.TestCase):
    """La ropa no es una maleta y una bicicleta no pasa a ser moto por un cuadro mal detectado."""

    def pasar(self, secuencia, cfg=None):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f, salida = FiltroObjetos(), []
        for i, (clase, conf) in enumerate(secuencia):
            salida = f.filtrar([Objeto(7, clase, np.array([0, 0, 50, 50.0]), conf)], i * 0.4, cfg)
        return salida

    def test_ropa_con_poca_confianza_no_se_vuelve_maleta(self):
        self.assertEqual(self.pasar([("maleta", 0.42)] * 8), [])

    def test_una_maleta_clara_si_se_confirma(self):
        self.assertEqual([o.clase for o in self.pasar([("maleta", 0.8)] * 4)], ["maleta"])

    def test_la_clase_es_la_mas_votada_no_la_ultima(self):
        sec = [("motocicleta", 0.8)] * 6 + [("bicicleta", 0.5)]
        self.assertEqual(self.pasar(sec)[0].clase, "motocicleta")

    def test_si_duda_entre_bici_y_moto_lo_dice(self):
        sec = [("bicicleta", 0.6), ("motocicleta", 0.6)] * 3
        self.assertEqual(self.pasar(sec)[0].clase, "bicicleta o moto")

    def test_clase_apagada_en_ajustes(self):
        self.assertEqual(self.pasar([("maleta", 0.9)] * 4, {"clases": {"maleta": False}}), [])

    def test_la_lista_repetida_del_detector_no_cuenta_como_otra_deteccion(self):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f, o = FiltroObjetos(), Objeto(3, "mochila", np.array([0, 0, 40, 40.0]), 0.9)
        for i in range(6):                       # el mismo objeto devuelto en 6 cuadros sin detección nueva
            salida = f.filtrar([o], i * 0.1)
        self.assertEqual(salida, [])


class VocabularioAbierto(unittest.TestCase):
    """Detector que reconoce objetos por su nombre (YOLO-World). Se prueba con un YOLO simulado."""

    def detector(self, nombres, detecciones, tmp):
        import types
        from vigia import modelos

        class Arr(np.ndarray):          # imita los tensores de ultralytics: .cpu().numpy(), .int(), .tolist()
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
            def int(self): return Arr.de(np.asarray(self).astype(int))
            def tolist(self): return np.asarray(self).tolist()
            @staticmethod
            def de(a): return np.asarray(a).view(Arr)

        class Res:
            def __init__(self, filas):
                self.keypoints = None
                if not filas:
                    self.boxes = types.SimpleNamespace(id=None)
                    return
                self.boxes = types.SimpleNamespace(
                    xyxy=Arr.de([f[2] for f in filas]).astype(float).view(Arr), id=Arr.de([f[0] for f in filas]),
                    cls=Arr.de([f[1] for f in filas]), conf=Arr.de([f[3] for f in filas]).astype(float).view(Arr))

        class YOLOFalso:
            def __init__(self, ruta):
                self.ruta, self.names = ruta, ({i: n for i, n in enumerate(nombres)} if "vocab" in ruta else {})
            def track(self, frame, **kw):
                self.kw = kw
                return [Res([] if "pose" in self.ruta else detecciones)]
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=YOLOFalso)
        try:
            ruta = Path(tmp) / "vigia-vocabulario.pt"
            ruta.write_bytes(b"x")
            cfg = dict(CFG["modelos"], objetos_abierto=str(ruta), pose="pose.pt")
            det = modelos.DetectorYOLO(cfg)
            return det, det.procesar(np.zeros((10, 10, 3), np.uint8), 0.0)
        finally:
            del sys.modules["ultralytics"]

    def test_nombres_en_espanol_y_sin_personas(self):
        with tempfile.TemporaryDirectory() as tmp:
            det, obs = self.detector(["person", "clothes", "sack", "tablet"],
                                     [(1, 0, [0, 0, 5, 5], 0.9), (2, 1, [1, 1, 4, 4], 0.4),
                                      (3, 2, [2, 2, 6, 6], 0.5), (4, 3, [3, 3, 7, 7], 0.6)], tmp)
        self.assertTrue(det.abierto)
        self.assertEqual(sorted(o.clase for o in obs.objetos), ["ropa", "saco", "tablet"])
        self.assertIsNone(det.obj.kw["classes"])                  # busca todo su vocabulario
        from vigia.modelos import SEGUIDOR_OBJETOS
        self.assertEqual(det.obj.kw["tracker"], SEGUIDOR_OBJETOS)  # seguidor que acepta confianza baja
        self.assertEqual(det.obj.kw["imgsz"], CFG["modelos"]["tamano_imagen_objetos"])

    def test_sin_el_archivo_usa_coco(self):
        from vigia.modelos import CLASES_OBJETO, modelo_abierto, nombres_objetos
        cfg = dict(CFG["modelos"], objetos_abierto="models/no-existe.pt")
        self.assertIsNone(modelo_abierto(cfg))
        self.assertEqual(set(nombres_objetos(cfg)), set(CLASES_OBJETO.values()))

    def test_las_casillas_de_ajustes_siguen_al_vocabulario(self):
        from vigia.modelos import nombres_objetos
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "v.pt"
            ruta.write_bytes(b"x")
            nombres = nombres_objetos(dict(CFG["modelos"], objetos_abierto=str(ruta)))
        self.assertIn("ropa", nombres)
        self.assertIn("saco", nombres)
        self.assertNotIn("persona", nombres)

    def test_la_ropa_puesta_no_es_un_objeto(self):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f, g = FiltroObjetos(abierto=True), FiltroObjetos(abierto=True)    # un filtro por escena, como una cámara
        persona = Persona(1, np.array([100, 100, 200, 400.0]))
        for i in range(5):
            puesta = f.filtrar([Objeto(9, "ropa", np.array([110, 150, 190, 300.0]), 0.6)], i * 0.4, {}, [persona])
            colgada = g.filtrar([Objeto(8, "ropa", np.array([400, 100, 480, 250.0]), 0.6)], i * 0.4, {}, [persona])
        self.assertEqual(puesta, [])
        self.assertEqual([o.clase for o in colgada], ["ropa"])


# --------------------------------------------------------------------------- pruebas en clase (noviembre 2026)
def _con_munecas_tapadas(pid, k):
    """En un forcejeo las manos quedan detrás del cuerpo del otro: la pose no las ve (confianza baja)."""
    from vigia.simulador import _caja
    conf = np.full(17, 0.9)
    conf[[9, 10]] = 0.1
    return Persona(pid, _caja(k), k, conf)


class ForcejeoYDerribo(unittest.TestCase):
    """Video de la clase: dos personas forcejean cuerpo a cuerpo sin golpes rápidos y una termina en el suelo.
    Antes solo se detectaba la caída; la regla de violencia pedía golpes a más de 1.8 alturas/s."""

    def escena(self, x_a, x_b, segundos=6.0, cae_desde=None):
        from vigia.simulador import H

        def cuadros(t):
            a = esqueleto(x_a(t), 480, "quieto", t, 0, 1, 0)
            if cae_desde is not None and t >= cae_desde:
                prog = min(1.0, (t - cae_desde) / 0.7)
                b = esqueleto(x_b(cae_desde), 480, "cae" if prog < 1 else "tendido", t, 0, 1, prog)
            else:
                b = esqueleto(x_b(t), 480, "quieto", t, 0, -1, 0)
            return [_con_munecas_tapadas(1, a), _con_munecas_tapadas(2, b)], []
        return recorrer_escena(cuadros, segundos)

    def test_forcejeo_sin_golpes_es_violencia(self):
        from vigia.simulador import H
        ev = self.escena(lambda t: 400 + 0.06 * H * math.sin(2 * math.pi * 1.2 * t),
                         lambda t: 400 + 0.08 * H - 0.06 * H * math.sin(2 * math.pi * 1.2 * t), segundos=4)
        v = [e for e in ev if e.tipo == "violencia"]
        self.assertTrue(v and "forcejeo" in v[0].mensaje)

    def test_abrazo_quieto_no_es_violencia(self):
        from vigia.simulador import H
        ev = self.escena(lambda t: 400.0, lambda t: 400 + 0.08 * H + 0.004 * H * math.sin(9 * t), segundos=6)
        self.assertFalse([e for e in ev if e.tipo == "violencia"])

    def test_caminar_juntos_no_es_violencia(self):
        from vigia.simulador import H

        def cuadros(t):
            x = 200 + 0.7 * H * t
            a = esqueleto(x, 480, "camina", t, x, 1, 0)
            b = esqueleto(x + 0.1 * H, 485, "camina", t, x, 1, 0)
            return [persona(1, a), persona(2, b)], []
        self.assertFalse([e for e in recorrer_escena(cuadros, 3.5) if e.tipo == "violencia"])

    def test_derribo_tras_el_forcejeo(self):
        from vigia.simulador import H
        ev = self.escena(lambda t: 400 + 0.06 * H * math.sin(2 * math.pi * 1.2 * t),
                         lambda t: 400 + 0.08 * H - 0.06 * H * math.sin(2 * math.pi * 1.2 * t),
                         segundos=7, cae_desde=3.0)
        self.assertIn("caida", [e.tipo for e in ev])
        derribo = [e for e in ev if e.clave.startswith("violencia:derribo")]
        self.assertTrue(derribo and "#1 derribó a #2" in derribo[0].mensaje)

    def test_caerse_solo_no_es_agresion(self):
        def cuadros(t):
            prog = min(1.0, max(0.0, (t - 3.0) / 0.7))
            b = esqueleto(400, 480, "quieto" if t < 3 else ("cae" if prog < 1 else "tendido"), t, 0, 1, prog)
            a = esqueleto(800, 480, "quieto", t, 0, -1, 0)
            return [persona(1, a), persona(2, b)], []
        ev = recorrer_escena(cuadros, 7)
        self.assertIn("caida", [e.tipo for e in ev])
        self.assertFalse([e for e in ev if e.tipo == "violencia"])


class Arrebato(unittest.TestCase):
    """Videos de la clase: le quitan la mochila a alguien que la lleva puesta o en la mano. La regla de hurto
    solo vigilaba objetos que quedaron quietos, así que estos casos nunca daban alerta."""

    def escena(self, mochila_de, segundos=9.0, camara=lambda t: 0.0, oid_tras_tiron=None, clase="mochila"):
        """mochila_de(t) -> 'a', 'b', 'ambos' o None; la víctima #1 queda quieta en x=400 y el ladrón #2 llega
        desde la derecha y huye hacia la derecha. camara(t) desplaza toda la imagen (un celular que sigue al ladrón)."""
        from vigia.simulador import H
        from vigia.tipos import Objeto

        def x_ladron(t):
            if t < 4:
                return 820 - 50 * t
            if t < 6.5:
                return 400 + 0.45 * H
            return 400 + 0.45 * H + 2.2 * H * (t - 6.5)

        def cuadros(t):
            dx = -camara(t)
            xa, xb = 400 + dx, x_ladron(t) + dx
            a = esqueleto(xa, 480, "quieto", t, 0, 1, 0)
            b = esqueleto(xb, 480, "golpea" if 4 <= t < 6.5 else ("quieto" if t < 4 else "camina"), t, xb, -1, 0)
            pa, pb = persona(1, a), persona(2, b)
            quien = mochila_de(t)
            objetos = []
            if quien == "a" and clase == "ropa":   # una casaca en la mano de la víctima
                w = pa.punto(9)
                c = np.array([w[0] - 0.08 * H, w[1] - 0.02 * H, w[0] + 0.08 * H, w[1] + 0.30 * H])
            elif quien == "a":                     # a la espalda de la víctima
                c = np.array([xa - 0.10 * H, 480 - 0.78 * H, xa + 0.03 * H, 480 - 0.50 * H])
            elif quien == "ambos":                 # entre los dos, sujetada por las manos de ambos
                m = (pa.punto(9) + pb.punto(10)) / 2 if pb.punto(10) is not None else pa.punto(9)
                c = np.array([m[0] - 0.07 * H, m[1] - 0.10 * H, m[0] + 0.07 * H, m[1] + 0.10 * H])
            elif quien == "b":                     # en la mano del ladrón
                w = pb.punto(9)
                c = np.array([w[0] - 0.07 * H, w[1], w[0] + 0.07 * H, w[1] + 0.2 * H])
            if quien:
                oid = oid_tras_tiron if (oid_tras_tiron and quien == "b") else 70
                objetos.append(Objeto(oid, clase, c))
            return [pa, pb], objetos
        return [e for e in recorrer_escena(cuadros, segundos) if e.tipo == "hurto"]

    @staticmethod
    def tiron(t):
        return "a" if t < 5 else ("ambos" if t < 6.5 else "b")

    def test_le_quitan_la_mochila_que_lleva_puesta(self):
        h = self.escena(self.tiron)
        self.assertTrue(h, "el arrebato no generó alerta")
        self.assertIn("arrebato", h[0].mensaje)
        self.assertIn("#2 le quitó una mochila a la persona #1", h[0].mensaje)
        self.assertEqual(h[0].severidad, "alta")           # hubo forcejeo

    def test_la_mochila_cambia_de_numero_en_el_tiron(self):
        self.assertTrue(self.escena(self.tiron, oid_tras_tiron=71))

    def test_con_la_camara_siguiendo_al_ladron(self):
        from vigia.simulador import H
        sigue = lambda t: 0.0 if t < 6.5 else 2.2 * H * (t - 6.5)      # el ladrón queda quieto en la imagen
        self.assertTrue(self.escena(self.tiron, camara=sigue))

    def test_se_la_muestra_y_se_la_devuelve(self):
        self.assertEqual(self.escena(lambda t: "a" if t < 5 or t >= 6.5 else "ambos"), [])

    def test_la_lleva_su_dueno_todo_el_tiempo(self):
        self.assertEqual(self.escena(lambda t: "a"), [])


class HuidaConCamaraEnMovimiento(unittest.TestCase):
    def test_toca_y_huye_aunque_la_camara_lo_siga(self):
        from vigia.simulador import H

        def cuadros(t):
            pan = 0.0 if t < 1.5 else 2.5 * H * (t - 1.5)                 # el celular sigue al que huye
            if t < 1.5:
                a = esqueleto(400, 480, "golpea", t, 0, 1, 0)
            else:
                a = esqueleto(400 - 2.5 * H * (t - 1.5) + pan, 480, "camina", t, t * 300, -1, 0)
            v = esqueleto(400 + 0.6 * H + pan, 480, "quieto", t, 0, -1, 0)
            return [persona(1, a), persona(2, v)], []
        ev = [e for e in recorrer_escena(cuadros, 4.0) if e.tipo == "robo_huida"]
        self.assertTrue(ev and "#1 tocó a #2" in ev[0].mensaje)


class MochilaPuestaAlArrancar(unittest.TestCase):
    def test_no_es_parte_del_escenario(self):
        from vigia.simulador import H
        from vigia.tipos import Objeto
        reglas, hist, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), 0.0
        while t < 3:
            k = esqueleto(400, 480, "quieto", t, 0, 1, 0)
            mochila = Objeto(5, "mochila", np.array([400 - 0.10 * H, 480 - 0.78 * H, 400 + 0.03 * H, 480 - 0.50 * H]))
            obs = Observacion(t=t, personas=[persona(1, k)], objetos=[mochila])
            hist.actualizar(obs)
            reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        self.assertEqual(hist.objetos[5].estado["origen"], "nuevo")
        self.assertEqual(hist.objetos[5].estado["dueno"], 1)


class BorrarHistorial(unittest.TestCase):
    """En clase faltaba un botón para empezar de cero entre una prueba y otra."""

    def test_borra_alertas_fotos_y_enfriamiento(self):
        with tempfile.TemporaryDirectory() as d:
            g = GestorAlertas(Path(d), enfriamiento=30, agrupar=20)
            frame = np.zeros((60, 80, 3), np.uint8)
            ev = Evento("hurto", "alta", "m", "x 1", "hurto:1:2", [[1, 1, 9, 9]])
            g.procesar("cam1", [ev], frame)
            self.assertEqual(len(list((Path(d) / "capturas").glob("*.jpg"))), 1)
            self.assertEqual(g.borrar_todo(), 1)
            self.assertEqual(g.listar(), [])
            self.assertEqual(list((Path(d) / "capturas").glob("*.jpg")), [])
            nueva = g.procesar("cam1", [ev], frame)          # la misma alerta vuelve a salir enseguida
            self.assertEqual(nueva[0]["id"], 1)
            g.cerrar()

    def test_el_sistema_olvida_lo_aprendido(self):
        from vigia.servidor import Sistema
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(CFG)
            cfg["camaras"] = []
            s = Sistema(cfg, Path(d), demo=True)
            try:
                t = s.trabajadores["demo"]
                s._descartar("demo", "hurto", [[600, 430, 640, 470]])
                self.assertTrue((Path(d) / "ignorar.json").exists())
                reglas_antes = t.reglas
                s.reiniciar()
                self.assertFalse((Path(d) / "ignorar.json").exists())
                import time as _t
                for _ in range(50):                          # lo aplica el hilo de la cámara en su próximo cuadro
                    if t.reglas is not reglas_antes:
                        break
                    _t.sleep(0.05)
                self.assertIsNot(t.reglas, reglas_antes)
                self.assertEqual(t.reglas.ignorar, [])
            finally:
                for t in s.trabajadores.values():
                    t.detener()
                s.alertas.cerrar()
                s.encuesta.cerrar()

    def test_en_feria_cada_intento_es_su_propia_alerta(self):
        from vigia.servidor import Sistema
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(CFG)
            cfg["camaras"] = []
            s = Sistema(cfg, Path(d), demo=False, feria=True)
            self.assertEqual(s.alertas.agrupar, 0)
            s.alertas.cerrar()
            s.encuesta.cerrar()


class Prendas(unittest.TestCase):
    """En clase se probó con una prenda y no la detectaba: el filtro la tomaba por ropa puesta apenas quedaba
    delante del cuerpo, y las reglas de hurto no vigilaban ropa."""

    def test_prenda_en_la_mano_no_es_ropa_puesta(self):
        from vigia.objetos import FiltroObjetos
        from vigia.simulador import H
        from vigia.tipos import Objeto
        p = persona(1, esqueleto(400, 480, "quieto", 0, 0, 1, 0))
        w = p.punto(9)
        en_mano = np.array([w[0] - 0.08 * H, w[1] - 0.02 * H, w[0] + 0.08 * H, w[1] + 0.25 * H])
        puesta = np.array([400 - 0.13 * H, 480 - 0.82 * H, 400 + 0.13 * H, 480 - 0.48 * H])   # hombros a cadera
        f, salida = FiltroObjetos(abierto=True), {}
        for i in range(5):
            for nombre, caja, oid in (("mano", en_mano, 1), ("puesta", puesta, 2)):
                salida[nombre] = f.filtrar([Objeto(oid, "ropa", caja, 0.5)], i * 0.4, {}, [p])
        self.assertEqual([o.clase for o in salida["mano"]], ["ropa"])
        self.assertEqual(salida["puesta"], [])

    def test_le_quitan_la_casaca_de_la_mano(self):
        h = Arrebato().escena(Arrebato.tiron, clase="ropa")
        self.assertTrue(h and "le quitó una ropa" in h[0].mensaje)

    def test_ropa_de_la_tienda_con_clientes_delante_no_alerta(self):
        from vigia.simulador import H
        from vigia.tipos import Objeto
        reglas, hist, tipos, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), [], 0.0
        perchero = np.array([380.0, 480 - 0.70 * H, 440, 480 - 0.30 * H])
        while t < 60:
            x = 410 if t < 20 else 410 + (t - 20) * 120         # un cliente mira el perchero y se va
            per = [persona(1, esqueleto(x, 490, "quieto" if t < 20 else "camina", t, x, 1, 0))] if x < 950 else []
            obs = Observacion(t=t, personas=per, objetos=[Objeto(3, "ropa", perchero)] if t >= 12 else [])
            hist.actualizar(obs)
            tipos += [e.tipo for e in reglas.evaluar(obs, hist, ANCHO, ALTO, [])]
            t += 0.1
        self.assertNotIn("abandono", tipos)
        self.assertNotIn("hurto", tipos)

    def test_casaca_dejada_en_una_silla_y_otro_se_la_lleva(self):
        from vigia.simulador import H
        from vigia.tipos import Objeto
        reglas, hist, eventos, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), [], 0.0
        silla = np.array([270.0, 480 - 0.45 * H, 330, 480 - 0.20 * H])
        while t < 26:
            per, obj = [], []
            if t < 11:                                    # llega caminando con la casaca en la mano
                x = 150 + 15 * t
                p = persona(1, esqueleto(x, 480, "camina", t, x, 1, 0))
                w = p.punto(9)
                per.append(p)
                if t >= 4:
                    obj.append(Objeto(30, "ropa", np.array([w[0] - 0.08 * H, w[1], w[0] + 0.08 * H, w[1] + 0.3 * H])))
            elif t < 16:                                  # la deja en la silla y se va
                x = 315 + (t - 11) * 110
                per.append(persona(1, esqueleto(x, 480, "camina", t, x, 1, 0)))
                obj.append(Objeto(30, "ropa", silla))
            elif t < 19.5:                                # se acerca otra persona
                x = 800 - (t - 16) * 140
                per.append(persona(2, esqueleto(x, 480, "camina", t, x, -1, 0)))
                obj.append(Objeto(30, "ropa", silla))
            else:                                         # la toma y se va
                x = 310 - (t - 19.5) * 120
                p = persona(2, esqueleto(x, 480, "camina", t, x, -1, 0))
                w = p.punto(10)
                per.append(p)
                obj.append(Objeto(30, "ropa", np.array([w[0] - 0.08 * H, w[1], w[0] + 0.08 * H, w[1] + 0.3 * H])))
            obs = Observacion(t=t, personas=per, objetos=obj)
            hist.actualizar(obs)
            eventos += reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        h = [e for e in eventos if e.tipo == "hurto"]
        self.assertTrue(h and "#2" in h[0].mensaje, [e.mensaje for e in eventos])



class ObjetosSinParpadeo(unittest.TestCase):
    """En las pruebas la mochila aparecía y desaparecía: el detector la perdía un instante, le daba otro número y
    el filtro volvía a esperar 3 detecciones antes de mostrarla."""

    def test_cambio_de_numero_hereda_la_confirmacion(self):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f = FiltroObjetos()
        caja = np.array([300, 200, 360, 280.0])
        for i in range(4):
            f.filtrar([Objeto(1, "mochila", caja + i, 0.5)], i * 0.3, {})
        salida = f.filtrar([Objeto(7, "mochila", caja + 6, 0.35)], 1.5, {})     # mismo lugar, número nuevo
        self.assertEqual([(o.id, o.clase) for o in salida], [(7, "mochila")])

    def test_una_pasada_fallida_no_la_borra(self):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f = FiltroObjetos()
        caja = np.array([300, 200, 360, 280.0])
        for i in range(4):
            f.filtrar([Objeto(1, "mochila", caja, 0.5)], i * 0.3, {})
        self.assertEqual([o.id for o in f.filtrar([], 1.2, {})], [1])           # 0.3 s sin verla: sigue
        self.assertEqual(f.filtrar([], 2.5, {}), [])                              # 1.6 s: ya no

    def test_lo_que_nunca_se_confirmo_no_se_retiene(self):
        from vigia.objetos import FiltroObjetos
        from vigia.tipos import Objeto
        f = FiltroObjetos()
        f.filtrar([Objeto(1, "maleta", np.array([300, 200, 360, 280.0]), 0.3)], 0.0, {})
        self.assertEqual(f.filtrar([], 0.3, {}), [])

    def test_el_detector_usa_su_seguidor_y_tamano(self):
        from vigia.modelos import SEGUIDOR_OBJETOS
        import yaml as _y
        conf = _y.safe_load(Path(SEGUIDOR_OBJETOS).read_text(encoding="utf-8"))
        self.assertLess(conf["new_track_thresh"], 0.25)
        self.assertGreaterEqual(CFG["modelos"]["tamano_imagen_objetos"], CFG["modelos"]["tamano_imagen"])



class DuenoEnPantalla(unittest.TestCase):
    def test_el_objeto_muestra_a_su_dueno(self):
        h = Arrebato()
        from vigia.simulador import H
        from vigia.tipos import Objeto
        reglas, hist, t = Reglas(copy.deepcopy(CFG["reglas"])), Historial(), 0.0
        while t < 3:
            k = esqueleto(400, 480, "quieto", t, 0, 1, 0)
            m = Objeto(5, "mochila", np.array([400 - 0.10 * H, 480 - 0.78 * H, 400 + 0.03 * H, 480 - 0.50 * H]))
            obs = Observacion(t=t, personas=[persona(3, k)], objetos=[m])
            hist.actualizar(obs)
            reglas.evaluar(obs, hist, ANCHO, ALTO, [])
            t += 0.1
        self.assertEqual(reglas.duenos(hist), {5: 3})


class PausaDeVideo(unittest.TestCase):
    """Pedido después de clase: poder pausar un video subido para explicar lo que está pasando."""

    def _sistema(self, d):
        from vigia.servidor import Sistema
        cfg = copy.deepcopy(CFG)
        cfg["camaras"] = []
        return Sistema(cfg, Path(d), demo=True)

    def _esperar(self, cond, maximo=5.0):
        import time as _t
        fin = _t.time() + maximo
        while _t.time() < fin and not cond():
            _t.sleep(0.05)
        return cond()

    def test_pausa_congela_imagen_y_reloj(self):
        import time as _t
        with tempfile.TemporaryDirectory() as d:
            s = self._sistema(d)
            try:
                t = s.trabajadores["demo"]
                self.assertTrue(self._esperar(lambda: t.jpeg is not None and t.estado.get("pausable")))
                self.assertTrue(t.pausar(True))
                self.assertTrue(t.estado["pausado"])
                _t.sleep(0.3)                                # un cuadro que estuviera en curso termina
                imagen, reloj = t.jpeg, t._reloj()
                _t.sleep(0.6)
                self.assertIs(t.jpeg, imagen)                # no se analiza ni se dibuja nada nuevo
                self.assertAlmostEqual(t._reloj(), reloj, delta=0.01)   # el reloj del análisis está detenido
                self.assertTrue(t.pausar(False))
                self.assertFalse(t.estado["pausado"])
                self.assertGreaterEqual(t._pausa_total, 0.8)  # el tiempo en pausa no cuenta para las reglas
                self.assertTrue(self._esperar(lambda: t.jpeg is not imagen))
            finally:
                for t in s.trabajadores.values():
                    t.detener()
                s.alertas.cerrar()

    def test_camara_en_vivo_no_se_pausa(self):
        from vigia.camara import Trabajador

        class LectorVivo:
            finito, pausado, terminado = False, False, False

            def detener(self):
                pass

        with tempfile.TemporaryDirectory() as d:
            s = self._sistema(d)
            try:
                t = s.trabajadores["demo"]
                t.demo, t.lector = False, LectorVivo()       # como una cámara RTSP o una webcam
                self.assertFalse(t.pausable)
                self.assertFalse(t.pausar(True))
                self.assertIsNone(t._pausa_desde)
                t.lector.finito = True                       # un video subido sí
                self.assertTrue(t.pausar(True))
                self.assertTrue(t.lector.pausado)
                t.pausar(False)
                self.assertFalse(t.lector.pausado)
                self.assertIsInstance(t, Trabajador)
            finally:
                for t in s.trabajadores.values():
                    t.detener()
                s.alertas.cerrar()

    def test_ruta_de_pausa(self):
        """La ruta se llama como lo haría el navegador (ASGI directo: no hace falta instalar httpx)."""
        import asyncio
        import json as _json
        from vigia.servidor import crear_app

        async def pedir(app, ruta, cuerpo=b""):
            enviado, respuesta = False, {"cuerpo": b""}

            async def recibir():
                nonlocal enviado
                if enviado:
                    await asyncio.sleep(3600)
                enviado = True
                return {"type": "http.request", "body": cuerpo, "more_body": False}

            async def enviar(m):
                if m["type"] == "http.response.start":
                    respuesta["estado"] = m["status"]
                elif m["type"] == "http.response.body":
                    respuesta["cuerpo"] += m.get("body", b"")
            alcance = {"type": "http", "method": "POST", "path": ruta, "raw_path": ruta.encode(), "query_string": b"",
                       "headers": [(b"content-type", b"application/json"), (b"host", b"localhost")],
                       "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000), "scheme": "http",
                       "root_path": "", "http_version": "1.1", "asgi": {"version": "3.0"}}
            await app(alcance, recibir, enviar)
            return respuesta["estado"], _json.loads(respuesta["cuerpo"] or b"{}")

        async def probar(d):
            cfg = copy.deepcopy(CFG)
            cfg["camaras"] = []
            app = crear_app(cfg, Path(d), demo=True)
            async with app.router.lifespan_context(app):
                sis = app.state.sistema
                t = sis.trabajadores["demo"]
                for _ in range(100):
                    if t.estado.get("pausable"):
                        break
                    await asyncio.sleep(0.05)
                try:
                    estado, r = await pedir(app, "/api/camaras/demo/pausa", b'{"pausado": true}')
                    self.assertEqual((estado, r["pausado"]), (200, True))
                    self.assertTrue(t.estado["pausado"])
                    estado, r = await pedir(app, "/api/camaras/demo/pausa")      # sin cuerpo: alterna
                    self.assertEqual((estado, r["pausado"]), (200, False))
                    estado, _ = await pedir(app, "/api/camaras/nada/pausa", b"{}")
                    self.assertEqual(estado, 404)
                finally:
                    sis.alertas.cerrar()

        with tempfile.TemporaryDirectory() as d:
            asyncio.run(probar(d))


class VideoSubidoACompas(unittest.TestCase):
    """En clase, dos videos de 9 y 13 s no daban ninguna alerta: la PC analizaba 1.3 cuadros por segundo mientras el
    video corría a su velocidad real, así que el tirón de la mochila (unos 2 s) caía entre dos cuadros y la mochila
    nunca juntaba las 3 detecciones que pide el filtro. Ahora el video avanza al ritmo del análisis."""

    FPS = 60

    def _video(self, d, segundos=2.0):
        import cv2
        ruta = str(Path(d) / "prueba.mp4")
        w = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (160, 120))
        for i in range(int(segundos * self.FPS)):
            f = np.zeros((120, 160, 3), np.uint8)
            f[:, :] = i % 250                                 # el brillo dice qué cuadro es
            w.write(f)
        w.release()
        return ruta

    def _esperar(self, cond, maximo=5.0):
        import time as _t
        fin = _t.time() + maximo
        while _t.time() < fin and not cond():
            _t.sleep(0.01)
        return cond()

    def _tomar(self, lector):
        caja = {}
        self.assertTrue(self._esperar(lambda: caja.setdefault("x", lector.tomar()) is not None
                                      or caja.pop("x") is not None))
        return caja["x"]

    def test_entrega_diez_cuadros_por_segundo_de_video_sin_saltarse_ninguno(self):
        import time as _t
        from vigia.camara import Lector
        with tempfile.TemporaryDirectory() as d:
            lector = Lector(self._video(d), fps_analisis=10)
            try:
                tiempos = []
                for _ in range(5):
                    _, t = self._tomar(lector)
                    tiempos.append(t)
                    _t.sleep(0.3)                             # un análisis lento (como en la PC de clase)
                pasos = np.diff(tiempos)
                self.assertTrue(np.allclose(pasos, 0.1, atol=0.02), pasos)   # 0.1 s de video por cuadro, siempre
                self.assertAlmostEqual(lector.paso_video, 0.1, places=2)
            finally:
                lector.detener()

    def test_al_terminar_queda_quieto_y_se_puede_repetir(self):
        from vigia.camara import Lector
        with tempfile.TemporaryDirectory() as d:
            lector = Lector(self._video(d, segundos=1.0), fps_analisis=10)
            try:
                ultimo = 0.0
                while not lector.terminado:
                    r = lector.tomar()
                    if r is not None:
                        ultimo = r[1]
                    self._esperar(lambda: lector.terminado or not lector._tomado, 2.0)
                self.assertAlmostEqual(ultimo, 1.0, delta=0.15)
                self.assertIsNone(lector.tomar())             # no vuelve a empezar solo
                lector.rebobinar()
                _, t = self._tomar(lector)
                self.assertEqual(lector.vuelta, 1)
                self.assertGreater(t, ultimo)                 # el reloj del análisis nunca va hacia atrás
            finally:
                lector.detener()

    def test_el_analisis_usa_el_segundo_del_video(self):
        """El trabajador toma la hora del video (no la del reloj) y, si el video vuelve a empezar, olvida la vuelta
        anterior pero conserva las falsas alarmas marcadas por el operador."""
        from vigia.camara import Lector
        from vigia.servidor import Sistema

        class Detector:
            vistos: list = []
            cada_n = None
            reinicios = 0

            def procesar(self, frame, t):
                self.vistos.append(t)
                return Observacion(t=t)

            def reiniciar_seguimiento(self):
                self.reinicios += 1

        class SinRostros:
            disponible = False

        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(CFG)
            cfg["camaras"] = []
            s = Sistema(cfg, Path(d), demo=True)
            t = s.trabajadores["demo"]
            try:
                t._parar = True                               # se maneja a mano, cuadro por cuadro
                t.demo, t.sim = False, None
                t.detector, t.rostros = Detector(), SinRostros()
                t.lector = Lector(self._video(d, segundos=1.0), fps_analisis=10)
                t.reglas.ignorar = [[1, 1, 5, 5]]
                for _ in range(4):
                    self._esperar(lambda: not t.lector._tomado)
                    t._paso()
                self.assertTrue(np.allclose(t.detector.vistos, [0.1, 0.2, 0.3, 0.4], atol=0.02), t.detector.vistos)
                self.assertEqual(t.detector.cada_n, 2)
                self.assertIsNotNone(t.estado["velocidad"])
                t.lector.rebobinar()
                self.assertTrue(self._esperar(lambda: t.lector.vuelta == 1 and not t.lector._tomado))
                t._paso()
                self.assertEqual(t.detector.reinicios, 1)
                self.assertEqual(t.reglas.ignorar, [[1, 1, 5, 5]])
            finally:
                if t.lector:
                    t.lector.detener()
                s.alertas.cerrar()
