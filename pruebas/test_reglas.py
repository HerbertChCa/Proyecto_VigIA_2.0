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
