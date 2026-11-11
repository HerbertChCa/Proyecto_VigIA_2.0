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

    def test_tiron_fallido_no_es_hurto(self):
        import estres
        ev = estres.correr(lambda: estres.tiron_fallido, 11, (), estres.Condicion(), copy.deepcopy(CFG["reglas"]))
        self.assertNotIn("hurto", {e.tipo for e in ev})


if __name__ == "__main__":
    unittest.main()
