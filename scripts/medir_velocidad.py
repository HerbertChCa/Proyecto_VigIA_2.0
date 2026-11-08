"""Mide cuántos cuadros por segundo analiza esta computadora y en qué se va el tiempo de cada cuadro.

La precisión en vivo depende de esa cifra: un tirón de mochila o un golpe duran 1 a 2 s, y las reglas necesitan
verlos en varios cuadros. Con 7 o más cuadros por segundo, el análisis en vivo se acerca al de un video en
cámara lenta («precisión alta» en la tarjeta de la cámara).

Uso (con un video de prueba o la webcam):
  python scripts\\medir_velocidad.py VID_20261105_155429.mp4
  python scripts\\medir_velocidad.py VID_20261105_155429.mp4 --rapido        el perfil rápido de la feria
  python scripts\\medir_velocidad.py 0 --cuadros 100                          la webcam

Pruebe en este orden y vuelva a medir después de cada paso, para ver cuánto ganó:
  1. tal como está;   2. --rapido;   3. después de  python scripts\\acelerar_modelos.py  (y --rapido si lo usará).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.historial import Historial  # noqa: E402
from vigia.objetos import FiltroObjetos  # noqa: E402
from vigia.reglas import Reglas  # noqa: E402
from vigia.rendimiento import aplicar_perfil  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fuente", help="video o número de webcam")
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--rapido", action="store_true", help="medir con el perfil rápido")
    ap.add_argument("--cuadros", type=int, default=60)
    ap.add_argument("--camaras", type=int, default=1, help="cuántas cámaras tendrá a la vez (se turnan los modelos)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    perfil = aplicar_perfil(cfg, "rapido" if a.rapido else cfg.get("procesamiento", {}).get("perfil"))
    from vigia.modelos import AnalizadorRostros, DetectorYOLO, asignar_rostros
    m, proc, priv = cfg["modelos"], cfg["procesamiento"], cfg["privacidad"]
    det = DetectorYOLO(m)
    rostros = AnalizadorRostros(m["rostros"], m["emociones"])
    filtro, hist, reglas = FiltroObjetos(abierto=det.abierto), Historial(), Reglas(cfg["reglas"])
    usar_rostros = rostros.disponible and (priv["difuminar_rostros"] or priv["analizar_emociones"])

    cap = cv2.VideoCapture(int(a.fuente) if a.fuente.isdigit() else a.fuente)
    salto = 1
    if not a.fuente.isdigit():
        salto = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / proc["fps_objetivo"]))
    tiempos = {"personas": [], "objetos": [], "rostros": [], "reglas": []}
    total, n, i = [], 0, 0
    while n < a.cuadros + 3:
        ok, frame = cap.read()
        if not ok:
            if a.fuente.isdigit():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        i += 1
        if i % salto:
            continue
        if frame.shape[1] > proc["ancho_max"]:
            e = proc["ancho_max"] / frame.shape[1]
            frame = cv2.resize(frame, (proc["ancho_max"], int(frame.shape[0] * e)))
        t = n / proc["fps_objetivo"]
        t0 = time.perf_counter()
        corre_objetos = det.n % max(1, m["objetos_cada_n_frames"]) == 0
        obs = det.procesar(frame, t)
        t1 = time.perf_counter()
        if usar_rostros and n % max(1, int(proc.get("rostros_cada_n_frames", 1))) == 0:
            obs.rostros = rostros.detectar(frame, priv["analizar_emociones"], int(proc.get("ancho_rostros", 0)))
            asignar_rostros(obs.rostros, obs.personas)
        t2 = time.perf_counter()
        obs.objetos = filtro.filtrar(obs.objetos, t, cfg["reglas"]["objetos"], obs.personas)
        hist.actualizar(obs)
        reglas.evaluar(obs, hist, frame.shape[1], frame.shape[0], [])
        t3 = time.perf_counter()
        n += 1
        if n <= 3:                                   # los primeros cuadros cargan cosas: no cuentan
            continue
        tiempos["personas"].append(det.ms["personas"])
        if corre_objetos:
            tiempos["objetos"].append(det.ms["objetos"])
        tiempos["rostros"].append(1000 * (t2 - t1))
        tiempos["reglas"].append(1000 * (t3 - t2))
        total.append(1000 * (t3 - t0))
    if not total:
        print("No se pudo leer la fuente.")
        return

    ms = float(np.mean(total))
    fps = 1000 / ms / a.camaras
    cada_obj = max(1, m["objetos_cada_n_frames"])
    print(f"\n=== {len(total)} cuadros · perfil {perfil} · personas {m['tamano_imagen']} px en "
          f"{det.aceleracion['personas']}, objetos {m.get('tamano_imagen_objetos')} px en {det.aceleracion['objetos']} ===")
    filas = [("Personas y esqueleto (YOLO pose)", np.mean(tiempos["personas"]), "cada cuadro"),
             ("Objetos (mochila, ropa…)", np.mean(tiempos["objetos"] or [0]), f"1 de cada {cada_obj} cuadros"),
             ("Rostros y emociones", np.mean(tiempos["rostros"]), "promedio por cuadro"),
             ("Reglas", np.mean(tiempos["reglas"]), "cada cuadro")]
    for nombre, v, nota in filas:
        print(f"  {nombre:34s} {v:7.1f} ms   ({nota})")
    print(f"  {'Total por cuadro':34s} {ms:7.1f} ms")
    nivel = "ALTA" if fps >= 7 else "MEDIA" if fps >= 4 else "BAJA"
    print(f"\nCon {a.camaras} cámara(s): unos {fps:.1f} cuadros por segundo por cámara -> precisión en vivo {nivel}")
    print(f"(el panel pide {proc['fps_objetivo']}; por encima de eso no se gana nada)")

    consejos = []
    if det.aceleracion["personas"] == "cpu":
        consejos.append("Exporte los modelos a OpenVINO (2-3 veces más rápido en CPU): "
                        "python scripts\\acelerar_modelos.py" + (" --rapido" if perfil == "rapido" else ""))
    if perfil != "rapido":
        consejos.append("Pruebe el perfil rápido: python scripts\\medir_velocidad.py ... --rapido "
                        "(y en la feria: python -m vigia --rapido)")
    if np.mean(tiempos["rostros"]) > 0.25 * ms:
        consejos.append("Los rostros pesan mucho: apague las emociones (analizar_emociones: false) o suba "
                        "rostros_cada_n_frames")
    if a.camaras > 1:
        consejos.append("Cada cámara extra divide la velocidad: en la demostración deje solo la que se está usando")
    if fps < 7:
        print("\nPara subirla, en este orden:")
        for k, c in enumerate(consejos, 1):
            print(f"  {k}. {c}")


if __name__ == "__main__":
    main()
