"""Modelos de visión. Corren en CPU; si hay una tarjeta gráfica NVIDIA (CUDA) la usan solos.

- YOLO11n-pose: detecta personas, estima 17 puntos del cuerpo y las sigue entre frames (ByteTrack).
- YOLO11n: detecta mochilas, bolsos y maletas, también con seguimiento.
- YuNet + MobileFaceNet (OpenCV Zoo): rostros y expresión facial. Son opcionales.
"""
from __future__ import annotations

import logging
import os
import threading
import time
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

def ruta_acelerada(ruta: str) -> Optional[Path]:
    """Carpeta del modelo exportado a OpenVINO (scripts/acelerar_modelos.py), si existe: `yolo11n-pose.pt` ->
    `yolo11n-pose_openvino_model/`. En CPU Intel o AMD suele correr 2 a 3 veces más rápido que el .pt."""
    p = Path(ruta)
    candidatos = [p.with_name(p.stem + "_openvino_model")]
    if not p.is_absolute() and len(p.parts) == 1:      # solo el nombre (ultralytics lo bajó): se busca en models/
        candidatos.append(RAIZ / "models" / (p.stem + "_openvino_model"))
    for c in candidatos:
        if (c / "metadata.yaml").exists():
            return c
    return None


def _imgsz_exportado(carpeta: Path) -> Optional[int]:
    """Tamaño de imagen con el que se exportó (un modelo exportado solo acepta ese tamaño)."""
    try:
        import yaml
        datos = yaml.safe_load((carpeta / "metadata.yaml").read_text(encoding="utf-8")) or {}
        tam = datos.get("imgsz")
        return int(max(tam)) if isinstance(tam, (list, tuple)) else int(tam) if tam else None
    except Exception:  # noqa: BLE001 - un metadata raro: se usa el .pt
        return None


def dispositivo() -> str:
    """'0' (la primera tarjeta NVIDIA) si PyTorch la ve; si no, 'cpu'."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def cargar_yolo(ruta: str, tarea: str, imgsz: int, cfg: dict):
    """Carga un modelo YOLO de la forma más rápida disponible y devuelve (modelo, imgsz, cómo corre).

    1. Con tarjeta NVIDIA: el .pt en la GPU (en media precisión).
    2. Sin ella, si se exportó a OpenVINO (python scripts/acelerar_modelos.py): la versión exportada.
    3. Si no, el .pt en la CPU, como siempre.
    Si la versión exportada falla al cargar, se vuelve al .pt: acelerar nunca debe dejar a VigIA sin modelo."""
    from ultralytics import YOLO
    modo = str(cfg.get("aceleracion", "auto")).lower()      # auto | openvino | no
    disp = dispositivo()
    if modo != "no" and disp == "cpu":
        carpeta = ruta_acelerada(ruta)
        if carpeta is not None:
            tam = _imgsz_exportado(carpeta) or imgsz
            if tam != imgsz:
                log.warning("%s se exportó a %d px y config.yaml pide %d px: se usa %d. Para cambiarlo, vuelva a "
                            "correr scripts/acelerar_modelos.py.", carpeta.name, tam, imgsz, tam)
            try:
                return YOLO(str(carpeta), task=tarea), tam, "openvino"
            except Exception:  # noqa: BLE001
                log.exception("No se pudo cargar %s; se usa el modelo normal (.pt)", carpeta)
    modelo = YOLO(ruta)
    return modelo, imgsz, ("gpu" if disp != "cpu" else "cpu")


# Las cámaras se turnan para usar los modelos: correr varias inferencias a la vez en CPU no es más rápido,
# solo reparte los núcleos entre todas y deja sin CPU al servidor web.
_INFERENCIA = threading.Lock()


def _fijar_hilos():
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:  # noqa: BLE001 - sin torch no hay nada que fijar
        pass
    cv2.setNumThreads(1)


SEGUIDOR_OBJETOS = str(Path(__file__).resolve().parent / "seguidores" / "objetos.yaml")

CLASES_OBJETO = {24: "mochila", 26: "bolso", 28: "maleta", 1: "bicicleta", 2: "auto", 3: "motocicleta",
                 43: "cuchillo", 34: "bate", 63: "laptop", 67: "celular"}

# Vocabulario abierto (YOLO-World): el modelo busca objetos por su nombre escrito, gracias a un codificador de
# texto tipo CLIP, en vez de limitarse a las 80 clases de COCO (que no incluye «ropa» ni «saco»). Los nombres
# van en inglés porque es lo que entiende el modelo; la etiqueta que se muestra es la española.
VOCABULARIO_BASE = {
    "person": "persona",              # absorbe a las personas para que su ropa puesta no se lea como «ropa»
    "backpack": "mochila", "handbag": "bolso", "suitcase": "maleta", "sack": "saco", "plastic bag": "bolsa",
    "cardboard box": "caja", "clothes": "ropa", "jacket": "ropa", "sweater": "ropa", "hoodie": "ropa",
    "coat": "ropa", "laptop": "laptop", "cell phone": "celular", "tablet": "tablet",
    "camera": "cámara", "headphones": "audífonos", "wallet": "billetera",
    "bicycle": "bicicleta", "motorcycle": "motocicleta", "car": "auto", "knife": "cuchillo", "baseball bat": "bate",
}


def vocabulario(cfg: dict) -> dict:
    return cfg.get("vocabulario") or VOCABULARIO_BASE


def modelo_abierto(cfg: dict) -> Optional[str]:
    """Ruta del detector de vocabulario abierto si está preparado y activado; si no, se usa el de COCO."""
    if not cfg.get("vocabulario_abierto", True) or not cfg.get("objetos_abierto"):
        return None
    p = Path(cfg["objetos_abierto"])
    p = p if p.is_absolute() else RAIZ / p
    return str(p) if p.exists() else None


def nombres_objetos(cfg: dict) -> list[str]:
    """Clases que puede reportar el detector de objetos en uso (para las casillas de Ajustes)."""
    if modelo_abierto(cfg):
        return [n for n in dict.fromkeys(vocabulario(cfg).values()) if n != "persona"]
    return list(dict.fromkeys(CLASES_OBJETO.values()))


class DetectorYOLO:
    """Un detector por cámara: el seguidor de YOLO guarda estado propio por instancia."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        _fijar_hilos()
        self.disp = dispositivo()
        self.media = self.disp != "cpu"          # media precisión en GPU: casi el doble de rápido
        self.pose, self.imgsz_pose, modo_pose = cargar_yolo(ruta_modelo(cfg["pose"]), "pose", cfg["tamano_imagen"], cfg)
        abierto = modelo_abierto(cfg)
        tam_obj = cfg.get("tamano_imagen_objetos", cfg["tamano_imagen"])
        if abierto:
            self.obj, self.imgsz_obj, modo_obj = cargar_yolo(abierto, "detect", tam_obj, cfg)
            vocab = vocabulario(cfg)
            self._mapa = {int(i): vocab.get(n, n) for i, n in self.obj.names.items()}
            self._clases, self._conf = None, float(cfg.get("confianza_objetos_abierto", 0.15))
            log.info("Detector de objetos de vocabulario abierto: %s", ", ".join(sorted(set(self._mapa.values()))))
        else:
            self.obj, self.imgsz_obj, modo_obj = cargar_yolo(ruta_modelo(cfg["objetos"]), "detect", tam_obj, cfg)
            # umbral bajo a propósito: decide el filtro de objetos (confianza media y varias detecciones seguidas)
            self._mapa, self._clases, self._conf = CLASES_OBJETO, list(CLASES_OBJETO), 0.20
        self.abierto = abierto is not None
        self.aceleracion = {"personas": modo_pose, "objetos": modo_obj}
        log.info("Modelos: personas en %s (%d px), objetos en %s (%d px)", modo_pose, self.imgsz_pose, modo_obj,
                 self.imgsz_obj)
        self.ms = {"personas": 0.0, "objetos": 0.0}   # milisegundos de la última pasada de cada modelo
        self.n = 0
        self.cada_n: int | None = None      # si se fija, reemplaza objetos_cada_n_frames (videos subidos: 2)
        self._ultimos_objetos: list[Objeto] = []

    def reiniciar_seguimiento(self) -> None:
        """El video volvió a empezar: los seguidores olvidan a las personas y objetos de la vuelta anterior."""
        for m in (self.pose, self.obj):
            for tr in getattr(getattr(m, "predictor", None), "trackers", None) or []:
                try:
                    tr.reset()
                except Exception:           # versión de ultralytics sin reset(): se sigue igual
                    log.debug("No se pudo reiniciar el seguidor", exc_info=True)
        self.n = 0
        self._ultimos_objetos = []

    def procesar(self, frame: np.ndarray, t: float) -> Observacion:
        c = self.cfg
        obs = Observacion(t=t)
        t0 = time.perf_counter()
        with _INFERENCIA:
            r = self.pose.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz_pose,
                                conf=c["confianza"], classes=[0], verbose=False, device=self.disp, half=self.media)[0]
        self.ms["personas"] = 1000 * (time.perf_counter() - t0)
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
        if self.n % max(1, self.cada_n or c["objetos_cada_n_frames"]) == 0:
            # los objetos son más chicos que las personas: se buscan en una imagen más grande, con un seguidor
            # que acepta detecciones de baja confianza (una mochila puesta o una prenda en la mano)
            t0 = time.perf_counter()
            with _INFERENCIA:
                ro = self.obj.track(frame, persist=True, tracker=SEGUIDOR_OBJETOS, imgsz=self.imgsz_obj,
                                    conf=self._conf, classes=self._clases, verbose=False,
                                    device=self.disp, half=self.media)[0]
            self.ms["objetos"] = 1000 * (time.perf_counter() - t0)
            self._ultimos_objetos = []
            if ro.boxes is not None and ro.boxes.id is not None:
                for caja, oid, cls, cf in zip(ro.boxes.xyxy.cpu().numpy(), ro.boxes.id.int().cpu().tolist(),
                                              ro.boxes.cls.int().cpu().tolist(), ro.boxes.conf.cpu().numpy()):
                    nombre = self._mapa.get(cls, "objeto")
                    if nombre != "persona":           # las personas las sigue el modelo de pose
                        self._ultimos_objetos.append(Objeto(int(oid), nombre, caja, float(cf)))
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

    def detectar(self, frame: np.ndarray, emociones: bool, ancho_max: int = 0) -> list[Rostro]:
        """`ancho_max`: YuNet busca los rostros en una copia reducida a ese ancho (antes recorría la imagen completa
        en cada cuadro, lo que en una laptop costaba casi tanto como la pose). Las cajas y las marcas se devuelven
        en coordenadas de la imagen original, y la expresión se lee del rostro a resolución completa."""
        if self.det is None:
            return []
        h, w = frame.shape[:2]
        esc = ancho_max / w if 0 < ancho_max < w else 1.0
        chica = cv2.resize(frame, (int(w * esc), int(h * esc))) if esc < 1 else frame
        self.det.setInputSize((chica.shape[1], chica.shape[0]))
        _, caras = self.det.detect(chica)
        out = []
        if caras is None:
            return out
        for f in caras:
            f = np.asarray(f, dtype=np.float32).copy()
            f[:14] /= esc                                   # caja (x, y, ancho, alto) y 5 marcas, a tamaño original
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
