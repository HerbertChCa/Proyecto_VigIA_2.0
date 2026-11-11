"""Videos de prueba de la noche antes de la feria (arrebato y «toca y huye» que terminan fuera del cuadro)."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_reglas import CFG  # noqa: E402


class VideosDeLaFeria(unittest.TestCase):
    """Videos de prueba de la noche antes de la feria: el ladrón sale del cuadro en menos de un segundo, borroso."""

    def test_arrebato_que_termina_fuera_de_la_camara(self):
        import estres
        ev = estres.correr(lambda: estres.arrebato_sale, 11, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertTrue(any("fuera de la cámara" in e.mensaje for e in ev), [e.mensaje for e in ev])

    def test_toca_y_sale_corriendo_del_cuadro(self):
        import estres
        ev = estres.correr(lambda: estres.toca_y_sale, 6, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertIn("robo_huida", {e.tipo for e in ev})

    def test_saludar_e_irse_caminando_no_es_robo(self):
        import estres
        ev = estres.correr(lambda: estres.saluda_y_se_va, 8, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertFalse({"robo_huida", "hurto", "violencia"} & {e.tipo for e in ev})

    def test_mochila_perdida_en_el_tiron_y_ladron_sale_por_el_borde(self):
        """Video real 11.08.50: YOLO deja de ver la mochila (negra, borrosa) 2.7 s antes de que el ladrón salga."""
        import estres

        def escena(t):
            ps, objs = estres.arrebato_sale(t)
            xb = 400 + (t - 8.5) * 2.8 * estres.H
            if t >= 9.2 and xb < estres.ANCHO + 20:          # sigue corriendo hasta salir por el borde derecho
                ps.append(estres.T.persona(2, estres.esqueleto(xb, 470, "camina", t, xb, 1, 0)))
            return ps, (objs if t < 6.6 else [])          # la mochila se pierde en pleno forcejeo
        ev = estres.correr(lambda: escena, 11, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertTrue(any("fuera de la cámara" in e.mensaje for e in ev), [e.mensaje for e in ev])

    def test_mochila_perdida_y_el_otro_desaparece_en_medio_no_es_hurto(self):
        """Si quien forcejeaba deja de verse en medio de la imagen (tapado), no se supone que se la llevó."""
        import estres

        def escena(t):
            ps, objs = estres.arrebato_sale(t)
            if t >= 9.0:
                ps = ps[:1]                                # #2 desaparece lejos de los bordes, 1.5 s después
            return ps, (objs if t < 6.6 else [])
        ev = estres.correr(lambda: escena, 11, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertNotIn("hurto", {e.tipo for e in ev})

    def test_tiron_fallido_no_es_hurto(self):
        import estres
        ev = estres.correr(lambda: estres.tiron_fallido, 11, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertNotIn("hurto", {e.tipo for e in ev})


if __name__ == "__main__":
    unittest.main()
