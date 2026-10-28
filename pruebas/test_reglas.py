"""Pruebas de las reglas y del gestor de alertas. Ejecutar: python -m unittest discover pruebas"""
import copy
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
