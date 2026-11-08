"""Diagnóstico de un video: muestra, segundo a segundo, lo que VigIA ve y por qué una regla dispara o no.

Sirve para entender una prueba que falló (por ejemplo, «le quitamos la mochila y no dio alerta»). Pasa el video
por el mismo flujo que una cámara en vivo (detector, filtro de objetos, historial y reglas) y escribe:

  diagnostico/<video>_anotado.mp4   el video con esqueletos, objetos, zonas y las alertas en el momento en que saltan
  diagnostico/<video>_cuadros.csv   una fila por cuadro analizado con las medidas que usan las reglas
  y un resumen en pantalla: qué detectó, con qué confianza, y qué le faltó a cada regla.

Uso:
  python scripts\\diagnosticar_video.py VID_20261105_155016.mp4
  python scripts\\diagnosticar_video.py prueba.mp4 --zonas prueba.zonas.json --fps 10
  python scripts\\diagnosticar_video.py noticia.mp4 --desde 17 --hasta 29     (un tramo sin cortes de edición)

Las zonas usan el mismo formato que datos/zonas.json (lista de {"nombre", "tipo", "puntos"} de 0 a 1).
Grabe las pruebas con el celular quieto (apoyado o en un trípode): si la cámara se mueve, las zonas dibujadas
dejan de coincidir con el piso y las velocidades de las personas en la imagen dejan de ser reales.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.dibujo import dibujar  # noqa: E402
from vigia.historial import Historial  # noqa: E402
from vigia.objetos import FiltroObjetos  # noqa: E402
from vigia.reglas import PORTATILES, Reglas, angulo_tronco, esta_tendido, medidas_tendido, sostiene  # noqa: E402


def movimiento_de_camara(previo, actual) -> float:
    """Desplazamiento medio del fondo entre dos cuadros, en píxeles (0 = cámara quieta)."""
    if previo is None:
        return 0.0
    a = cv2.cvtColor(cv2.resize(previo, (320, 180)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(actual, (320, 180)), cv2.COLOR_BGR2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    return float(np.hypot(dx, dy)) * previo.shape[1] / 320


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--zonas", default=None, help="archivo .json con zonas (opcional)")
    ap.add_argument("--fps", type=float, default=None, help="cuadros analizados por segundo (por defecto, config.yaml)")
    ap.add_argument("--salida", default="diagnostico")
    ap.add_argument("--desde", type=float, default=0.0, help="segundo del video donde empezar (tramos sin cortes)")
    ap.add_argument("--hasta", type=float, default=None, help="segundo del video donde terminar")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    from vigia.modelos import DetectorYOLO, modelo_abierto
    detector = DetectorYOLO(cfg["modelos"])
    abierto = modelo_abierto(cfg["modelos"])
    filtro = FiltroObjetos(abierto=abierto is not None)
    if abierto:
        print(f"Detector de objetos: vocabulario abierto ({abierto}); reconoce ropa, saco, bolsa y lo de config.yaml.")
    else:
        print("Detector de objetos: COCO (80 clases). NO reconoce ropa, saco ni bolsa: prepare el vocabulario con\n"
              "  pip install git+https://github.com/ultralytics/CLIP.git\n  python scripts\\preparar_vocabulario.py")
    reglas, hist = Reglas(cfg["reglas"]), Historial()
    zonas = json.loads(Path(a.zonas).read_text(encoding="utf-8")) if a.zonas else []
    for i, z in enumerate(zonas):
        z.setdefault("id", f"z{i}")

    ruta = Path(a.video)
    salida = Path(a.salida)
    salida.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ruta))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    if not 1 <= fps_video <= 240:
        fps_video = 30
    fps = a.fps or cfg["procesamiento"]["fps_objetivo"]
    paso = max(1, round(fps_video / fps))
    ancho_max = cfg["procesamiento"]["ancho_max"]

    escritor, previo, i = None, None, int(round(a.desde * fps_video))
    if i:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    filas, eventos, alertas_vistas = [], [], set()
    crudos = defaultdict(list)                 # clase -> confianzas de lo que el detector vio
    confirmados = Counter()                    # clase -> cuadros en que pasó el filtro
    vel_max = defaultdict(float)
    juntos_cuadros, portes, mov_cam, dos_o_mas, ids_vistos = 0, Counter(), [], 0, set()
    tiempos = []
    while True:
        ok, frame = cap.read()
        if not ok or (a.hasta is not None and i / fps_video > a.hasta):
            break
        if i % paso:
            i += 1
            continue
        if frame.shape[1] > ancho_max:
            e = ancho_max / frame.shape[1]
            frame = cv2.resize(frame, (ancho_max, int(frame.shape[0] * e)))
        t = i / fps_video
        i += 1
        mov = movimiento_de_camara(previo, frame)
        previo = frame.copy()
        mov_cam.append(mov)
        t0 = time.perf_counter()
        obs = detector.procesar(frame, t)
        tiempos.append(time.perf_counter() - t0)
        nuevos = [o for o in obs.objetos if not getattr(o, "_filtrado", False)]
        for o in nuevos:
            crudos[o.clase].append(o.conf)
        obs.objetos = filtro.filtrar(obs.objetos, t, cfg["reglas"]["objetos"], obs.personas)
        for o in obs.objetos:
            confirmados[o.clase] += 1
        hist.actualizar(obs)
        ids_vistos |= {p.id for p in obs.personas}
        h, w = frame.shape[:2]
        ev = reglas.evaluar(obs, hist, w, h, zonas)
        for e in ev:
            if e.clave not in alertas_vistas:
                alertas_vistas.add(e.clave)
                eventos.append((t, e))
        for pid, (_, v) in reglas._rap.items():
            vel_max[pid] = max(vel_max[pid], v)
        juntos = sum(1 for k, m in reglas._juntos.items() if m and m[-1][0] == t and m[-1][1])
        juntos_cuadros += bool(juntos)
        lleva = []
        for o in obs.objetos:
            if o.clase in PORTATILES:
                q = [p.id for p in obs.personas if sostiene(p, o.caja, clase=o.clase)]
                lleva.append(f"{o.clase}#{o.id}:{'+'.join(map(str, q)) or '-'}")
                portes[len(q)] += 1
        dos_o_mas += len(obs.personas) >= 2
        num = lambda x: "-" if x is None else f"{x:.2f}"
        postura = []                           # para entender una alerta de «persona en el suelo» o de caída
        for p in obs.personas:
            pista = hist.personas.get(p.id)
            if pista is None or not pista.muestras:
                continue
            alto_p = pista.alto_de_pie(t, 6.0)
            cc = cfg["reglas"]["caida"]
            d = medidas_tendido(pista.muestras[-1], alto_p, cc.get("puntos_cabeza_min", 3))
            marca = "TENDIDO" if esta_tendido(pista.muestras[-1], cc["angulo_horizontal"], alto_p,
                                              cc.get("puntos_cabeza_min", 3)) else "de pie"
            m = pista.muestras[-1]
            ang = angulo_tronco(m)
            postura.append(
                f"#{p.id}:{marca} pies-cad={num(d['pies_cadera'])} cad-cab={num(d['cadera_cabeza'])} (cruda {num(d['cadera_cabeza_cruda'])}) "
                f"canilla={num(d['canilla'])} "
                f"conf[cabeza={num(d['conf_cabeza'])} ({d['n_cabeza']} pts) cadera={num(d['conf_cadera'])} "
                f"rodillas={num(d['conf_rodillas'])} tobillos={num(d['conf_tobillos'])}] "
                f"caja={m.caja[2] - m.caja[0]:.0f}x{m.caja[3] - m.caja[1]:.0f} alto_ref={d['alto']:.0f} "
                f"tronco_grados={'-' if ang is None else f'{ang:.0f}'}")
        filas.append({
            "t": round(t, 2), "personas": " ".join(f"#{p.id}" for p in obs.personas),
            "objetos_detector": " ".join(f"{o.clase}:{o.conf:.2f}" for o in nuevos),
            "objetos_confirmados": " ".join(f"{o.clase}#{o.id}" for o in obs.objetos),
            "quien_lleva_cada_objeto": " ".join(lleva),
            "vel_extremidades_max": " ".join(f"#{pid}:{v:.1f}" for pid, (_, v) in reglas._rap.items()),
            "postura": " | ".join(postura),
            "cuerpo_a_cuerpo": juntos, "mov_camara_px": round(mov, 1),
            "alertas": " | ".join(e.mensaje for e in ev)})
        en_alerta = {p.id for p in obs.personas for e in ev for c in e.cajas if np.allclose(p.caja, c)}
        img = dibujar(frame.copy(), obs, zonas, en_alerta, {}, f"{ruta.name}  t={t:5.1f}s  camara {mov:4.1f}px",
                      duenos=reglas.duenos(hist))
        for k, (te, e) in enumerate([x for x in eventos if t - x[0] <= 2.5][-3:]):
            cv2.putText(img, f"ALERTA {e.tipo}: {e.mensaje}"[:90].encode("ascii", "ignore").decode(),
                        (10, h - 15 - 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 235), 2, cv2.LINE_AA)
        if escritor is None:
            escritor = cv2.VideoWriter(str(salida / f"{ruta.stem}_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps_video / paso, (w, h))
        escritor.write(img)
    cap.release()
    if escritor:
        escritor.release()
    if not filas:
        print("No se pudo leer el video.")
        return
    with open(salida / f"{ruta.stem}_cuadros.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0]))
        wr.writeheader()
        wr.writerows(filas)

    n = len(filas)
    umbral = cfg["reglas"]["violencia"]["velocidad_extremidades"]
    print(f"\n=== {ruta.name}: {n} cuadros analizados a {fps:g} cuadros/s, {1000 * np.mean(tiempos):.0f} ms por cuadro ===")
    mc = float(np.median(mov_cam))
    print(f"Cámara: se movió {mc:.1f} px por cuadro en promedio"
          + ("  <- EN MOVIMIENTO: apoye el celular; zonas y velocidades dejan de ser fiables." if mc > 3 else " (quieta)."))
    print("\nObjetos (lo que vio el detector -> lo que confirmó el filtro):")
    for clase in sorted(set(crudos) | set(confirmados)):
        cs = crudos.get(clase, [])
        print(f"  {clase:12s} detectado en {len(cs):3d} pasadas, confianza media {np.mean(cs) if cs else 0:.2f} "
              f"(máx {max(cs) if cs else 0:.2f}); confirmado en {confirmados[clase]} de {n} cuadros")
    if not any(c in PORTATILES for c in crudos):
        print("  -> No se detectó ningún objeto portátil (mochila, bolso, ropa…): sin objeto no hay hurto ni "
              "arrebato posibles.")
    print(f"\nQuién lleva los objetos: {portes[1]} cuadros con una persona, {portes[2] + portes[3]} con dos o más "
          f"(forcejeo), {portes[0]} sin nadie.")
    print(f"Personas: dos o más a la vez en {dos_o_mas} de {n} cuadros; {len(ids_vistos)} números de seguimiento "
          f"distintos ({', '.join(f'#{i}' for i in sorted(ids_vistos))}).")
    if dos_o_mas < 0.5 * n:
        print("  -> En la mayoría de los cuadros se vio una sola persona: si estaban pegadas, el modelo las tomó como "
              "una sola y no hay pareja para medir una pelea. Grabe más de lado o desde más lejos.")
    if len(ids_vistos) > 2 * max(1, max((len(f["personas"].split()) for f in filas), default=1)):
        print("  -> Los números cambiaron muchas veces (personas tapadas o cámara en movimiento): las reglas pierden la "
              "historia de cada una.")
    print(f"Cuerpo a cuerpo: {juntos_cuadros} de {n} cuadros con dos personas juntas.")
    print(f"Velocidad máxima de extremidades (umbral de golpes {umbral} alturas/s): "
          + ", ".join(f"#{pid} {v:.1f}" for pid, v in sorted(vel_max.items(), key=lambda x: -x[1])[:6]))
    print("\nAlertas:")
    for t, e in eventos:
        print(f"  {t:6.1f} s  {e.tipo:12s} {e.mensaje}\n            {e.explicacion}")
    if not eventos:
        print("  ninguna")
    print(f"\nArchivos en {salida.resolve()}: {ruta.stem}_anotado.mp4 y {ruta.stem}_cuadros.csv")


if __name__ == "__main__":
    main()
