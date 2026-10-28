"""Descarga de videos desde páginas (YouTube, Facebook, TikTok…) con yt-dlp, para analizarlos como cámara.

Una página de YouTube no es un archivo de video: OpenCV no puede abrirla. yt-dlp obtiene el video y lo guarda en
datos/videos, igual que un video subido desde la PC; al quitar la cámara, el archivo se borra.

Es opcional: pip install -r requirements-enlaces.txt. Para YouTube, yt-dlp también necesita Deno
(un intérprete de JavaScript; en Windows: winget install DenoLand.Deno). Las condiciones de YouTube no permiten
descargar videos sin permiso del autor: úselo solo para pruebas internas y no distribuya los videos.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

log = logging.getLogger("vigia.descargas")

PAGINAS_DE_VIDEO = re.compile(
    r"(youtube\.com|youtu\.be|facebook\.com|fb\.watch|tiktok\.com|instagram\.com|x\.com|twitter\.com|vimeo\.com)", re.I)
DURACION_MAX = 20 * 60          # segundos: un video más largo se recorta antes en la PC
# Solo hace falta la imagen (sin audio), en H.264 hasta 720p, que OpenCV abre en cualquier PC sin unir pistas.
FORMATO = ("bv*[height<=720][vcodec^=avc1]/b[height<=720][vcodec^=avc1]/"
           "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/b[height<=720]/b")


def disponible() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


class Descargas:
    def __init__(self, carpeta: Path):
        self.carpeta = carpeta
        self.trabajos: dict[str, dict] = {}

    def iniciar(self, url: str, titulo: str, al_terminar: Callable[[str, str], dict]) -> str:
        """Empieza la descarga en segundo plano; al terminar llama a al_terminar(ruta, titulo) -> cámara."""
        tid = uuid.uuid4().hex[:8]
        self.trabajos[tid] = {"estado": "preparando", "progreso": 0.0, "titulo": titulo, "camara": None, "error": None}
        threading.Thread(target=self._descargar, args=(tid, url, titulo, al_terminar), daemon=True).start()
        return tid

    def estado(self, tid: str) -> dict | None:
        return self.trabajos.get(tid)

    def _descargar(self, tid, url, titulo, al_terminar) -> None:
        t = self.trabajos[tid]
        try:
            import yt_dlp
            self.carpeta.mkdir(parents=True, exist_ok=True)
            base = f"{datetime.now():%Y%m%d_%H%M%S}_enlace_{tid}"

            def avance(d):
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    if total:
                        t["progreso"] = round(100 * d.get("downloaded_bytes", 0) / total, 1)
                    t["estado"] = "descargando"

            opciones = {"format": FORMATO, "outtmpl": str(self.carpeta / f"{base}.%(ext)s"), "noplaylist": True,
                        "quiet": True, "no_warnings": True, "progress_hooks": [avance],
                        # certificados de Windows en lugar de los de certifi: así funciona también con antivirus que
                        # inspeccionan HTTPS (Avast, Kaspersky…) o proxies de la universidad
                        "compat_opts": {"no-certifi"}}
            with yt_dlp.YoutubeDL(opciones) as ydl:
                info = ydl.extract_info(url, download=False)
                dur = info.get("duration") or 0
                if dur > DURACION_MAX:
                    raise ValueError(f"El video dura {dur // 60} min; el límite es {DURACION_MAX // 60} min. "
                                     "Descárguelo, recorte el tramo que le interesa y súbalo desde su PC.")
                titulo = titulo or str(info.get("title") or "Video del enlace")[:40]
                t["titulo"] = titulo
                info = ydl.process_ie_result(info, download=True)
                pedidos = info.get("requested_downloads") or []
                ruta = Path(pedidos[0]["filepath"]) if pedidos and pedidos[0].get("filepath") \
                    else Path(ydl.prepare_filename(info))
            cap = cv2.VideoCapture(str(ruta))
            ok = cap.read()[0]
            cap.release()
            if not ok:
                ruta.unlink(missing_ok=True)
                raise ValueError("Se descargó el video, pero OpenCV no puede leer su formato.")
            t["camara"] = al_terminar(str(ruta), titulo)
            t["estado"], t["progreso"] = "lista", 100.0
        except Exception as e:                          # yt-dlp lanza sus propios errores (video privado, sin red…)
            log.warning("No se pudo descargar %s: %s", url, e)
            msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "")
            if "JavaScript runtime" in msg or "js runtime" in msg.lower():
                msg += " · Instale Deno (winget install DenoLand.Deno) y reinicie VigIA."
            t["estado"], t["error"] = "error", msg[:400]
