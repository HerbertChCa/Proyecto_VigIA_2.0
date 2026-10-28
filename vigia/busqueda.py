"""Búsqueda de capturas en lenguaje natural con CLIP (por ejemplo, «persona con mochila» o «moto de noche»).

Es la línea futura que propone el artículo: ninguno de los 44 estudios revisados usa modelos de
visión-lenguaje (Tabla 11). CLIP convierte cada captura y cada frase en un vector; la búsqueda devuelve las
capturas cuyo vector se parece más al de la frase. El modelo por defecto es multilingüe, así que entiende
consultas en español.

Es opcional: requiere `pip install -r requirements-busqueda.txt` y la primera vez descarga el modelo
(alrededor de 1.5 GB) desde Hugging Face. Solo indexa las capturas de alertas, que ya pasaron por el
pixelado de rostros o el modo solo esqueleto: no se graba ni se indexa video continuo.
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("vigia.busqueda")


class CodificadorCLIP:
    """Carga open_clip y convierte imágenes y textos en vectores normalizados."""

    def __init__(self, modelo: str, pesos: str):
        import open_clip
        import torch
        from PIL import Image
        self._torch, self._Image = torch, Image
        self.modelo, _, self.preparar = open_clip.create_model_and_transforms(modelo, pretrained=pesos)
        self.modelo.eval()
        self.tokenizar = open_clip.get_tokenizer(modelo)

    def imagen(self, ruta: Path) -> np.ndarray:
        with self._torch.no_grad():
            x = self.preparar(self._Image.open(ruta).convert("RGB")).unsqueeze(0)
            v = self.modelo.encode_image(x)[0].float().numpy()
        return v / (np.linalg.norm(v) + 1e-9)

    def texto(self, frase: str) -> np.ndarray:
        with self._torch.no_grad():
            v = self.modelo.encode_text(self.tokenizar([frase]))[0].float().numpy()
        return v / (np.linalg.norm(v) + 1e-9)


class Buscador:
    """Índice de capturas guardado en datos/indice_clip.npz; se actualiza en segundo plano."""

    def __init__(self, carpeta_capturas: Path, archivo_indice: Path,
                 fabrica: Optional[Callable[[], object]] = None):
        self.capturas = carpeta_capturas
        self.archivo = archivo_indice
        self._fabrica = fabrica
        self.codificador = None
        self.estado = "apagada"
        self._nombres: list[str] = []
        self._vectores: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._cola: queue.Queue = queue.Queue()
        self._hilo: Optional[threading.Thread] = None
        if self.archivo.exists():
            try:
                d = np.load(self.archivo, allow_pickle=False)
                self._nombres, self._vectores = list(d["nombres"]), d["vectores"]
            except Exception:                       # índice dañado: se reconstruye
                self._nombres, self._vectores = [], None

    # ---- ciclo de vida
    def iniciar(self, modelo: str, pesos: str) -> None:
        if self._hilo and self._hilo.is_alive():
            return
        fabrica = self._fabrica or (lambda: CodificadorCLIP(modelo, pesos))
        self.estado = "cargando el modelo (la primera vez se descarga)"
        self._hilo = threading.Thread(target=self._trabajar, args=(fabrica,), daemon=True)
        self._hilo.start()

    def _trabajar(self, fabrica) -> None:
        try:
            self.codificador = fabrica()
        except ImportError:
            self.estado = "falta instalar: pip install -r requirements-busqueda.txt"
            log.warning("Búsqueda desactivada: %s", self.estado)
            return
        except Exception as e:                     # sin internet, modelo inexistente, etc.
            self.estado = f"error al cargar el modelo: {e}"
            log.exception("No se pudo cargar CLIP")
            return
        # capturas que todavía no están en el índice (por ejemplo, las de antes de activar la búsqueda)
        conocidas = set(self._nombres)
        for f in sorted(self.capturas.glob("*.jpg")):
            if f.name not in conocidas:
                self._cola.put(f.name)
        self.estado = "lista"
        while True:
            nombre = self._cola.get()
            if nombre is None:
                return
            self._indexar(nombre)

    def agregar(self, nombre: str) -> None:
        """Se llama al guardar cada captura de alerta."""
        if self.estado != "apagada":
            self._cola.put(nombre)

    def detener(self) -> None:
        self._cola.put(None)

    # ---- índice
    def _indexar(self, nombre: str) -> None:
        ruta = self.capturas / nombre
        if not ruta.exists():
            return
        try:
            v = self.codificador.imagen(ruta).astype(np.float32)
        except Exception:
            log.exception("No se pudo indexar %s", nombre)
            return
        with self._lock:
            self._nombres.append(nombre)
            self._vectores = v[None] if self._vectores is None else np.vstack([self._vectores, v])
            np.savez(self.archivo, nombres=np.array(self._nombres), vectores=self._vectores)

    @property
    def total(self) -> int:
        return len(self._nombres)

    def pendientes(self) -> int:
        return self._cola.qsize()

    def buscar(self, frase: str, k: int = 12) -> list[dict]:
        if self.codificador is None or self.estado != "lista":
            raise RuntimeError(f"La búsqueda no está lista: {self.estado}")
        frase = frase.strip()
        if not frase:
            return []
        q = self.codificador.texto(frase).astype(np.float32)
        with self._lock:
            if self._vectores is None:
                return []
            puntajes = self._vectores @ q
            nombres = list(self._nombres)
        orden = np.argsort(-puntajes)
        out = []
        for i in orden:
            if (self.capturas / nombres[i]).exists():      # las capturas antiguas se borran por retención
                out.append({"captura": nombres[i], "puntaje": round(float(puntajes[i]), 4)})
            if len(out) >= k:
                break
        return out
