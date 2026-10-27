"""Anotaciones sobre el video y privacidad por diseño (difuminado de cabezas)."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .tipos import ESQUELETO, Observacion

COLOR_NORMAL = (180, 200, 60)     # BGR: verde azulado
COLOR_ALERTA = (60, 60, 235)      # rojo
COLOR_OBJETO = (40, 180, 240)     # ámbar
COLOR_ZONA = {"restringida": (70, 70, 220), "merodeo": (40, 170, 240)}


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _texto(img, txt, org, color, escala=0.5, fondo=True):
    txt = _ascii(txt)
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    x, y = int(org[0]), int(org[1])
    if fondo:
        cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + b), (20, 20, 20), -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def cajas_cabeza(obs: Observacion) -> list[np.ndarray]:
    """Rostros detectados; si no hay detector de rostros, la cabeza estimada con la pose."""
    cajas = [r.caja for r in obs.rostros]
    if cajas:
        return cajas
    for p in obs.personas:
        if p.puntos is None:
            continue
        pts = [p.punto(i) for i in range(5)]
        pts = [q for q in pts if q is not None]
        if not pts:
            continue
        c = np.mean(pts, axis=0)
        r = 0.09 * p.alto if len(pts) < 3 else max(0.07 * p.alto, 1.3 * float(np.ptp(np.array(pts)[:, 0])))
        cajas.append(np.array([c[0] - r, c[1] - 1.2 * r, c[0] + r, c[1] + r]))
    return cajas


def difuminar(img: np.ndarray, cajas: list[np.ndarray]) -> None:
    h, w = img.shape[:2]
    for c in cajas:
        x1, y1, x2, y2 = (int(v) for v in c)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = img[y1:y2, x1:x2]
        peq = cv2.resize(roi, (max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10)), interpolation=cv2.INTER_LINEAR)
        img[y1:y2, x1:x2] = cv2.resize(peq, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def dibujar(img: np.ndarray, obs: Observacion, zonas: list[dict], en_alerta: set[int],
            emociones: dict[int, str], encabezado: str) -> np.ndarray:
    h, w = img.shape[:2]
    capa = img.copy()
    for z in zonas:
        pts = (np.array(z["puntos"]) * [w, h]).astype(np.int32)
        col = COLOR_ZONA.get(z.get("tipo"), (200, 200, 200))
        cv2.fillPoly(capa, [pts], col)
    cv2.addWeighted(capa, 0.22, img, 0.78, 0, img)
    for z in zonas:
        pts = (np.array(z["puntos"]) * [w, h]).astype(np.int32)
        col = COLOR_ZONA.get(z.get("tipo"), (200, 200, 200))
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        _texto(img, f"{z['nombre']} ({z.get('tipo', '')})", pts[0] + [4, 18], col, 0.45)

    for p in obs.personas:
        col = COLOR_ALERTA if p.id in en_alerta else COLOR_NORMAL
        x1, y1, x2, y2 = (int(v) for v in p.caja)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if p.puntos is not None:
            for a, b in ESQUELETO:
                pa, pb = p.punto(a), p.punto(b)
                if pa is not None and pb is not None:
                    cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 1, cv2.LINE_AA)
        etiqueta = f"#{p.id}"
        if p.id in emociones:
            etiqueta += f" {emociones[p.id]}"
        _texto(img, etiqueta, (x1, y1 - 4), col)

    for o in obs.objetos:
        x1, y1, x2, y2 = (int(v) for v in o.caja)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OBJETO, 2)
        _texto(img, f"{o.clase} #{o.id}", (x1, y1 - 4), COLOR_OBJETO, 0.42)

    _texto(img, encabezado, (10, 22), (235, 235, 235), 0.55)
    return img
