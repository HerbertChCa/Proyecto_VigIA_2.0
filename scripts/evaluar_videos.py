"""Evalúa todas las reglas de VigIA con videos propios etiquetados y escribe una ficha para el informe.

Responde a dos brechas del artículo (Tabla 11): la metodológica (reportar precisión, sensibilidad y F1 juntos,
con la latencia y el hardware) y la de validación (medir con datos de fuera del laboratorio: aquí, videos
grabados o reunidos por el equipo). Complementa a evaluar_rwf2000.py, que solo mide la violencia.

Organice los videos así (el nombre de la carpeta es el tipo de alerta):

    evaluacion/
      caida/positivos/        videos donde alguien se cae
      caida/normales/         parecidos sin caída: agacharse, sentarse, atarse los zapatos
      violencia/positivos/    peleas
      violencia/normales/     abrazos, juegos, saludos
      robo_vehiculo/...       (cualquier tipo: abandono, hurto, robo_huida, vandalismo, arma, intrusion…)
      normales/               escenas cotidianas sin ningún incidente (para medir falsas alarmas por hora)

Recorte cada video para que empiece unos segundos antes del incidente; en «abandono», al menos 10 s antes, porque lo
que se ve en los primeros 10 s se considera parte del escenario (config.yaml: reglas.objetos.calentamiento).

Para intrusión o merodeo, ponga junto al video un archivo con el mismo nombre y extensión .zonas.json con las zonas
(las mismas que guarda el panel en datos/zonas.json: lista de {"nombre", "tipo", "puntos"} con coordenadas de 0 a 1).

Uso:
  python scripts/evaluar_videos.py --datos evaluacion
  python scripts/evaluar_videos.py --datos evaluacion --operador datos/alertas.db

Un video positivo cuenta como acierto si la regla dispara al menos una vez; uno normal cuenta como falsa alarma
si dispara. Con pocos videos las cifras varían mucho, por eso se reporta el intervalo de confianza del 95 %
(Wilson) junto a cada proporción.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.alertas import NOMBRES  # noqa: E402
from vigia.historial import Historial  # noqa: E402
from vigia.reglas import Reglas  # noqa: E402

EXTENSIONES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
# alertas que cuentan como acierto para cada carpeta (una caída también se confirma con «en el suelo»)
ACEPTA = {"caida": {"caida", "en_suelo"}, "hurto": {"hurto"}, "robo_vehiculo": {"robo_vehiculo"}}


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    texto = f"{cpu} ({os.cpu_count()} hilos), {platform.system()} {platform.release()}"
    try:
        import torch
        texto += f", GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ", sin GPU (solo CPU)"
    except Exception:
        pass
    return texto


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza del 95 % para una proporción k/n (Wilson; se comporta bien con n pequeño)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def videos_de(base: Path) -> list[dict]:
    """Recorre la carpeta y devuelve {ruta, tipo, etiqueta} (etiqueta: positivo | normal | escena)."""
    lista = []
    for carpeta in sorted(p for p in base.iterdir() if p.is_dir()):
        if carpeta.name == "normales":
            lista += [{"ruta": v, "tipo": None, "etiqueta": "escena"} for v in sorted(carpeta.iterdir())
                      if v.suffix.lower() in EXTENSIONES]
            continue
        if carpeta.name not in NOMBRES:
            print(f"Aviso: se omite la carpeta «{carpeta.name}»: no es un tipo de alerta ({', '.join(NOMBRES)})")
            continue
        for sub, etiqueta in (("positivos", "positivo"), ("normales", "normal")):
            d = carpeta / sub
            if d.is_dir():
                lista += [{"ruta": v, "tipo": carpeta.name, "etiqueta": etiqueta} for v in sorted(d.iterdir())
                          if v.suffix.lower() in EXTENSIONES]
    return lista


def zonas_de(ruta: Path) -> list[dict]:
    f = ruta.with_suffix(".zonas.json")
    if not f.exists():
        return []
    zonas = json.loads(f.read_text(encoding="utf-8"))
    for i, z in enumerate(zonas):
        z.setdefault("id", f"z{i}")
    return zonas


def analizar_video(ruta: Path, cfg: dict, detector, fps: float) -> dict:
    """Pasa el video por el mismo flujo que una cámara (detector, historial y reglas) a `fps` cuadros/s."""
    cap = cv2.VideoCapture(str(ruta))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    if not 1 <= fps_video <= 240:
        fps_video = 30
    paso = max(1, round(fps_video / fps))
    reglas, hist, zonas = Reglas(cfg["reglas"]), Historial(), zonas_de(ruta)
    eventos, vistos, i, t_mod, n_mod, ancho_max = [], set(), 0, 0.0, 0, cfg["procesamiento"]["ancho_max"]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % paso == 0:
            if frame.shape[1] > ancho_max:
                e = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, (ancho_max, int(frame.shape[0] * e)))
            t = i / fps_video
            t0 = time.perf_counter()
            obs = detector.procesar(frame, t)
            t_mod += time.perf_counter() - t0
            n_mod += 1
            hist.actualizar(obs)
            h, w = frame.shape[:2]
            for ev in reglas.evaluar(obs, hist, w, h, zonas):
                if ev.clave not in vistos:
                    vistos.add(ev.clave)
                    eventos.append((round(t, 1), ev.tipo))
        i += 1
    cap.release()
    return {"eventos": eventos, "duracion": i / fps_video, "ms_cuadro": 1000 * t_mod / n_mod if n_mod else 0.0}


def resumir(filas: list[dict]) -> dict:
    """Métricas por regla a partir de los resultados por video."""
    por_tipo = {}
    for tipo in sorted({f["tipo"] for f in filas if f["tipo"]}):
        pos = [f for f in filas if f["tipo"] == tipo and f["etiqueta"] == "positivo"]
        neg = [f for f in filas if f["tipo"] == tipo and f["etiqueta"] == "normal"]
        tp = sum(f["detectado"] for f in pos)
        fp = sum(f["detectado"] for f in neg)
        fn, tn = len(pos) - tp, len(neg) - fp
        prec = tp / (tp + fp) if tp + fp else None
        sens = tp / len(pos) if pos else None
        f1 = 2 * prec * sens / (prec + sens) if prec and sens else (0.0 if prec == 0 or sens == 0 else None)
        tiempos = sorted(f["segundos"] for f in pos if f["detectado"])
        por_tipo[tipo] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "sensibilidad": sens, "f1": f1,
            "especificidad": tn / len(neg) if neg else None,
            "ic_sensibilidad": wilson(tp, len(pos)), "ic_precision": wilson(tp, tp + fp),
            "mediana_s": tiempos[len(tiempos) // 2] if tiempos else None,
        }
    escenas = [f for f in filas if f["etiqueta"] == "escena"]
    horas = sum(f["duracion"] for f in escenas) / 3600
    falsas = sum(f["n_alertas"] for f in escenas)
    return {"por_tipo": por_tipo, "escenas": {"videos": len(escenas), "horas": horas, "alertas": falsas,
                                              "por_hora": falsas / horas if horas else None}}


def operador(db: Path) -> dict | None:
    """Precisión observada en uso real: veredictos del operador guardados por el panel."""
    if not db.exists():
        return None
    con = sqlite3.connect(db)
    filas = con.execute("SELECT tipo, SUM(veredicto = 'confirmada'), SUM(veredicto = 'falsa') FROM alertas "
                        "WHERE veredicto IS NOT NULL GROUP BY tipo").fetchall()
    con.close()
    return {t: (c or 0, f or 0) for t, c, f in filas}


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f} %"


def ic(par) -> str:
    return "—" if math.isnan(par[0]) else f"{100 * par[0]:.0f}–{100 * par[1]:.0f} %"


def ficha(res: dict, filas: list[dict], base: Path, cfg: dict, fps: float, ms: float, hw: str, op: dict | None) -> str:
    n_pos = sum(f["etiqueta"] == "positivo" for f in filas)
    n_neg = sum(f["etiqueta"] == "normal" for f in filas)
    l = ["# Ficha de evaluación de VigIA: todas las reglas con videos propios", "",
         f"Fecha: {date.today().isoformat()}. Conjunto: `{base}` ({len(filas)} videos: {n_pos} con el incidente, "
         f"{n_neg} parecidos sin incidente y {res['escenas']['videos']} de escenas cotidianas).", "",
         "| Regla | Videos (+/−) | Sensibilidad (IC 95 %) | Precisión (IC 95 %) | F1 | Especificidad | Detección (mediana) |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for tipo, m in res["por_tipo"].items():
        f1 = "—" if m["f1"] is None else f"{m['f1']:.2f}"
        demora = "—" if m["mediana_s"] is None else f"{m['mediana_s']} s"
        l.append(f"| {NOMBRES.get(tipo, tipo)} | {m['tp'] + m['fn']}/{m['fp'] + m['tn']} | {pct(m['sensibilidad'])} "
                 f"({ic(m['ic_sensibilidad'])}) | {pct(m['precision'])} ({ic(m['ic_precision'])}) | {f1} | "
                 f"{pct(m['especificidad'])} | {demora} |")
    e = res["escenas"]
    l += ["", f"Falsas alarmas en escenas cotidianas: {e['alertas']} alertas en {e['horas'] * 60:.1f} minutos de video"
          + (f" ({e['por_hora']:.1f} por hora)." if e["por_hora"] is not None else " (no hay videos en `normales/`).")]
    if op:
        l += ["", "## Precisión observada por el operador (uso real)", "",
              "| Regla | Confirmadas | Falsas alarmas | Precisión |", "| --- | --- | --- | --- |"]
        for tipo, (c, f) in sorted(op.items()):
            l.append(f"| {NOMBRES.get(tipo, tipo)} | {c} | {f} | {pct(c / (c + f)) if c + f else '—'} |")
    l += ["", "## Condiciones", "",
          f"- Latencia del análisis: {ms:.0f} ms por cuadro, a {fps:g} cuadros por segundo de video.",
          f"- Modelos: {cfg['modelos']['pose']} y {cfg['modelos']['objetos']}, entrada de {cfg['modelos']['tamano_imagen']} px; "
          f"cuadros reducidos a {cfg['procesamiento']['ancho_max']} px de ancho.",
          f"- Hardware: {hw}.",
          "- Un video positivo es un acierto si la regla dispara al menos una vez; uno parecido sin incidente es una "
          "falsa alarma si dispara. La sensibilidad es aciertos / videos positivos; la precisión, aciertos / disparos.",
          "- Con pocos videos por regla los intervalos son amplios: reporte las cifras con su intervalo y el número de videos."]
    return "\n".join(l) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datos", required=True, help="carpeta con una subcarpeta por tipo de alerta (ver arriba)")
    ap.add_argument("--fps", type=float, default=None, help="cuadros analizados por segundo (por defecto, config.yaml)")
    ap.add_argument("--operador", help="datos/alertas.db para agregar la precisión observada por el operador")
    ap.add_argument("--salida", default=str(RAIZ / "resultados_videos.csv"))
    ap.add_argument("--ficha", default=str(RAIZ / "ficha_videos.md"))
    a = ap.parse_args()

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    fps = a.fps or cfg["procesamiento"]["fps_objetivo"]
    base = Path(a.datos)
    lista = videos_de(base)
    if not lista:
        sys.exit(f"No hay videos en {base}. Revise la estructura de carpetas al inicio de este script.")

    from vigia.modelos import DetectorYOLO
    filas, latencias = [], []
    for n, v in enumerate(lista, 1):
        r = analizar_video(v["ruta"], cfg, DetectorYOLO(cfg["modelos"]), fps)   # detector nuevo: seguimiento limpio
        acepta = ACEPTA.get(v["tipo"], {v["tipo"]})
        primeras = [t for t, tipo in r["eventos"] if tipo in acepta]
        fila = {"video": str(v["ruta"].relative_to(base)), "tipo": v["tipo"], "etiqueta": v["etiqueta"],
                "detectado": bool(primeras), "segundos": primeras[0] if primeras else None,
                "n_alertas": len(r["eventos"]), "duracion": round(r["duracion"], 1),
                "alertas": "; ".join(f"{t}s {tipo}" for t, tipo in r["eventos"])}
        filas.append(fila)
        latencias.append(r["ms_cuadro"])
        marca = "detectado" if fila["detectado"] else "no detectado"
        print(f"[{n}/{len(lista)}] {fila['video']} ({v['etiqueta']}): "
              f"{marca if v['tipo'] else str(fila['n_alertas']) + ' alertas'}")

    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    res = resumir(filas)
    op = operador(Path(a.operador)) if a.operador else None
    texto = ficha(res, filas, base, cfg, fps, sum(latencias) / len(latencias), hardware(), op)
    Path(a.ficha).write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Detalle por video: {a.salida}\nFicha para el informe: {a.ficha}")


if __name__ == "__main__":
    main()
