"""Modelos de visión. Todo corre en CPU.

- YOLO11n-pose: detecta personas, estima 17 puntos del cuerpo y las sigue entre frames (ByteTrack).
- YOLO11n: detecta mochilas, bolsos y maletas, también con seguimiento.
- YuNet + MobileFaceNet (OpenCV Zoo): rostros y expresión facial. Son opcionales.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .tipos import EMOCIONES, Objeto, Observacion, Persona, Rostro

log = logging.getLogger("vigia.modelos")
RAIZ = Path(__file__).resolve().parent.parent


def ruta_modelo(ruta: str) -> str:
    """Ruta relativa a la carpeta del proyecto. Si un peso de YOLO no está, se usa su nombre
    y ultralytics lo descarga solo la primera vez."""
    p = Path(ruta)
    p = p if p.is_absolute() else RAIZ / p
    if p.exists() or p.suffix != ".pt":
        return str(p)
    return p.name

CLASES_OBJETO = {24: "mochila", 26: "bolso", 28: "maleta"}


class DetectorYOLO:
    """Un detector por cámara: el seguidor de YOLO guarda estado propio por instancia."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # se importa aquí para que el modo demo no lo necesite
        self.cfg = cfg
        self.pose = YOLO(ruta_modelo(cfg["pose"]))
        self.obj = YOLO(ruta_modelo(cfg["objetos"]))
        self.n = 0
        self._ultimos_objetos: list[Objeto] = []

    def procesar(self, frame: np.ndarray, t: float) -> Observacion:
        c = self.cfg
        obs = Observacion(t=t)
        r = self.pose.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=c["tamano_imagen"],
                            conf=c["confianza"], classes=[0], verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            cajas = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kcf = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None
            for i, pid in enumerate(ids):
                obs.personas.append(Persona(
                    id=int(pid), caja=cajas[i], conf=float(confs[i]),
                    puntos=None if kxy is None else kxy[i],
                    conf_puntos=None if kcf is None else kcf[i]))
        if self.n % max(1, c["objetos_cada_n_frames"]) == 0:
            ro = self.obj.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=c["tamano_imagen"],
                                conf=0.30, classes=list(CLASES_OBJETO), verbose=False)[0]
            self._ultimos_objetos = []
            if ro.boxes is not None and ro.boxes.id is not None:
                for caja, oid, cls, cf in zip(ro.boxes.xyxy.cpu().numpy(), ro.boxes.id.int().cpu().tolist(),
                                              ro.boxes.cls.int().cpu().tolist(), ro.boxes.conf.cpu().numpy()):
                    self._ultimos_objetos.append(Objeto(int(oid), CLASES_OBJETO.get(cls, "objeto"), caja, float(cf)))
        obs.objetos = list(self._ultimos_objetos)
        self.n += 1
        return obs


# Puntos de referencia de un rostro alineado de 112x112 (ojos, nariz, comisuras)
_REF_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class AnalizadorRostros:
    """Detecta rostros (YuNet) y, si se pide, su expresión (MobileFaceNet FER, 7 clases)."""

    def __init__(self, ruta_rostros: str, ruta_emociones: str):
        ruta_rostros, ruta_emociones = ruta_modelo(ruta_rostros), ruta_modelo(ruta_emociones)
        self.det = None
        self.fer = None
        if Path(ruta_rostros).exists():
            self.det = cv2.FaceDetectorYN.create(ruta_rostros, "", (320, 320), 0.7, 0.3, 50)
        else:
            log.warning("No se encontró %s: se difuminará la cabeza usando la pose.", ruta_rostros)
        if Path(ruta_emociones).exists():
            self.fer = cv2.dnn.readNet(ruta_emociones)
        else:
            log.warning("No se encontró %s: el análisis de emociones queda desactivado.", ruta_emociones)

    @property
    def disponible(self) -> bool:
        return self.det is not None

    def detectar(self, frame: np.ndarray, emociones: bool) -> list[Rostro]:
        if self.det is None:
            return []
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, caras = self.det.detect(frame)
        out = []
        if caras is None:
            return out
        for f in caras:
            x, y, cw, ch = f[:4]
            rostro = Rostro(caja=np.array([x, y, x + cw, y + ch], dtype=float))
            if emociones and self.fer is not None:
                rostro.probs = self._expresion(frame, f[4:14].reshape(5, 2))
            out.append(rostro)
        return out

    def _expresion(self, frame, marcas) -> Optional[np.ndarray]:
        m, _ = cv2.estimateAffinePartial2D(marcas.astype(np.float32), _REF_112, method=cv2.LMEDS)
        if m is None:
            return None
        cara = cv2.warpAffine(frame, m, (112, 112))
        cara = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cara = (cara - 0.5) / 0.5
        self.fer.setInput(cv2.dnn.blobFromImage(cara))
        logits = self.fer.forward().reshape(-1)[:len(EMOCIONES)]
        e = np.exp(logits - logits.max())
        return e / e.sum()


def asignar_rostros(rostros: list[Rostro], personas: list[Persona]) -> None:
    """Asocia cada rostro a la persona cuya parte superior de la caja lo contiene."""
    for r in rostros:
        cx, cy = (r.caja[0] + r.caja[2]) / 2, (r.caja[1] + r.caja[3]) / 2
        mejor, dist = None, 1e9
        for p in personas:
            x1, y1, x2, y2 = p.caja
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.45 * (y2 - y1):
                d = abs(cx - (x1 + x2) / 2)
                if d < dist:
                    mejor, dist = p.id, d
        r.persona_id = mejor
