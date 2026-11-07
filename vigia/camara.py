"""Lectura de cámaras (webcam, RTSP o archivo) y análisis en un hilo por cámara."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

from .alertas import GestorAlertas
from .dibujo import cajas_cabeza, difuminar, dibujar, solo_esqueleto
from .historial import Historial
from .objetos import FiltroObjetos
from .reglas import Reglas
from .tipos import Observacion

log = logging.getLogger("vigia.camara")


class Lector:
    """Lee frames en su propio hilo.

    Cámara en vivo (webcam, celular, RTSP): conserva solo el último cuadro, para no acumular retraso.
    Video con duración (archivo subido o enlace a un .mp4): avanza al ritmo del análisis, no del reloj. Entrega
    `fps_analisis` cuadros por cada segundo de video y espera a que el análisis tome uno antes de leer el siguiente.
    Si la computadora es lenta, el video se ve en cámara lenta, pero cada segundo de video se analiza completo
    (antes se reproducía a su velocidad real y, a 1-2 cuadros analizados por segundo, un tirón de mochila de dos
    segundos quedaba entre dos cuadros y la mochila nunca llegaba a confirmarse)."""

    def __init__(self, fuente, fps_analisis: float = 10, repetir: bool = False):
        self.fuente = int(fuente) if str(fuente).isdigit() else str(fuente)
        self.es_archivo = isinstance(self.fuente, str) and not self.fuente.lower().startswith(("rtsp", "http"))
        self.es_enlace = isinstance(self.fuente, str) and self.fuente.lower().startswith(("http://", "https://"))
        self.fps_analisis = max(1.0, float(fps_analisis))
        self.repetir = repetir
        self.frame: Optional[np.ndarray] = None
        self.conectada = False
        self.finito = self.es_archivo     # tiene duración (archivo o enlace a un .mp4): se puede pausar
        self.pausado = False
        self.terminado = False            # el video llegó al final (queda en el último cuadro)
        self.vuelta = 0                   # cuántas veces volvió a empezar: el análisis olvida lo anterior
        self.t_video = 0.0                # segundo del video del cuadro entregado (con las vueltas sumadas)
        self.paso_video = 1.0 / self.fps_analisis   # segundos de video entre dos cuadros entregados
        self._numero = 0                  # cuadros entregados
        self._tomado = True               # el análisis ya tomó el cuadro actual: se puede leer el siguiente
        self._rebobinar = False
        self._desfase = 0.0
        self._parar = False
        self._lock = threading.Lock()
        threading.Thread(target=self._bucle, daemon=True).start()

    def _abrir(self):
        if isinstance(self.fuente, int):
            cap = cv2.VideoCapture(self.fuente, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.fuente)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.fuente)
        elif not self.es_archivo:
            # celular (IP Webcam, DroidCam) o cámara IP: si no responde en 5 s se reintenta, en vez de esperar sin fin
            try:
                cap = cv2.VideoCapture(self.fuente, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            except (cv2.error, TypeError, AttributeError):    # OpenCV anterior a 4.6
                cap = cv2.VideoCapture(self.fuente)
        else:
            cap = cv2.VideoCapture(self.fuente)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _bucle(self):
        espera = 1.0
        while not self._parar:
            cap = self._abrir()
            if not cap.isOpened():
                self.conectada = False
                log.warning("No se pudo abrir %s; reintento en %.0f s", self.fuente, espera)
                time.sleep(espera)
                espera = min(espera * 2, 30)
                continue
            espera = 1.0
            fps_archivo = cap.get(cv2.CAP_PROP_FPS) or 25
            if not 1 <= fps_archivo <= 240:
                fps_archivo = 25
            # un enlace a un archivo (.mp4) tiene duración, como un archivo local;
            # una transmisión en vivo (.m3u8, RTSP) no tiene número de frames
            finito = self.es_archivo or (self.es_enlace and (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0)
            self.finito = finito
            if finito:
                self._leer_video(cap, fps_archivo)
            else:
                self._leer_en_vivo(cap)
            cap.release()
            self.conectada = False

    def _leer_en_vivo(self, cap):
        while not self._parar:
            ok, f = cap.read()
            if not ok:
                break                           # transmisión caída: se vuelve a abrir
            with self._lock:
                self.frame = f
                self._numero += 1
            self.conectada = True

    def _leer_video(self, cap, fps_archivo: float):
        salto = max(1, round(fps_archivo / self.fps_analisis))   # p. ej. un video de 60 fps: 1 de cada 6 cuadros
        self.paso_video = salto / fps_archivo
        leidos = 0
        while not self._parar:
            if self._rebobinar:
                self._rebobinar = False
                self._desfase = self.t_video + 1.0  # el reloj del análisis sigue avanzando: nunca vuelve atrás
                self.vuelta += 1
                self.terminado = False
                if not self.es_archivo:
                    return                          # un enlace se vuelve a abrir desde el principio
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                leidos = 0
            if self.pausado or self.terminado or not self._tomado:
                time.sleep(0.005)
                continue
            ok = True
            for _ in range(salto - 1):              # grab() salta cuadros sin convertirlos: más liviano que read()
                ok = cap.grab()
                if not ok:
                    break
                leidos += 1
            if ok:
                ok, f = cap.read()
            if not ok:
                if leidos == 0:
                    time.sleep(1)
                    return                          # no se pudo leer nada: se reintenta abrirlo
                if self.repetir:
                    self._rebobinar = True
                else:
                    self.terminado = True           # queda en el último cuadro hasta que se pida repetir
                continue
            leidos += 1
            with self._lock:
                self.frame = f
                self.t_video = self._desfase + leidos / fps_archivo
                self._numero += 1
                self._tomado = False
            self.conectada = True

    def ultimo(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def tomar(self) -> Optional[tuple[np.ndarray, float]]:
        """Video con duración: el cuadro siguiente y su segundo de video, o None si todavía no hay uno nuevo."""
        with self._lock:
            if self._tomado or self.frame is None:
                return None
            self._tomado = True
            return self.frame.copy(), self.t_video

    def rebobinar(self) -> None:
        self._rebobinar = True

    def detener(self):
        self._parar = True


class Trabajador:
    def __init__(self, cam: dict, cfg: dict, alertas: GestorAlertas, zonas: list[dict], demo: bool = False):
        self.id = cam["id"]
        self.nombre = cam.get("nombre", cam["id"])
        self.fuente = cam["fuente"]
        self.cfg = cfg
        self.alertas = alertas
        self.zonas = zonas
        self.demo = demo or str(self.fuente) == "demo"
        self.hist = Historial()
        from .modelos import modelo_abierto
        self.filtro = FiltroObjetos(abierto=modelo_abierto(cfg["modelos"]) is not None)
        self.reglas = Reglas(cfg["reglas"])
        self.jpeg: Optional[bytes] = None
        self.crudo: Optional[np.ndarray] = None   # último frame sin anotaciones (ya difuminado)
        self.estado = {"id": self.id, "nombre": self.nombre, "conectada": False, "fps": 0.0, "personas": 0,
                       "emociones": {}, "fuente": str(self.fuente), "error": None}
        self._en_alerta: dict[int, float] = {}
        self._tiempos: deque = deque(maxlen=20)
        self._parar = False
        self._reiniciar = False
        self._t0 = time.time()
        self._pausa_desde: Optional[float] = None   # instante en que se pausó (solo videos subidos y demo)
        self._pausa_total = 0.0                     # segundos en pausa acumulados: el reloj del análisis no avanza
        self._vuelta = 0                            # vuelta del video que se está analizando
        self.lector = None
        self.sim = None
        self.detector = None
        self.rostros = None
        threading.Thread(target=self._iniciar, daemon=True).start()

    # ---- arranque (la carga de modelos puede tardar la primera vez)
    def _iniciar(self):
        try:
            if self.demo:
                from .simulador import Simulador
                self.sim = Simulador()
            else:
                from .modelos import AnalizadorRostros, DetectorYOLO
                proc = self.cfg["procesamiento"]
                self.lector = Lector(self.fuente, proc["fps_objetivo"], bool(proc.get("repetir_videos", False)))
                self.detector = DetectorYOLO(self.cfg["modelos"])
                self.rostros = AnalizadorRostros(self.cfg["modelos"]["rostros"], self.cfg["modelos"]["emociones"])
        except Exception as e:  # p. ej. falta instalar ultralytics
            log.exception("No se pudo iniciar la cámara %s", self.id)
            self.estado["error"] = str(e)
            return
        self._bucle()

    def _bucle(self):
        periodo = 1.0 / self.cfg["procesamiento"]["fps_objetivo"]
        while not self._parar:
            inicio = time.time()
            if self._pausa_desde is not None:      # en pausa: se conserva la última imagen y no se analiza nada
                time.sleep(0.1)
                continue
            try:
                self._paso()
            except Exception as e:
                log.exception("Error procesando %s", self.id)
                self.estado["error"] = str(e)
                time.sleep(1)
            time.sleep(max(0.0, periodo - (time.time() - inicio)))

    @property
    def pausable(self) -> bool:
        """Solo se pausa lo que tiene duración (un video subido o un enlace a un archivo) y el modo demo; una cámara
        en vivo no se puede detener: lo que pasa mientras tanto se perdería."""
        return self.demo or (self.lector is not None and self.lector.finito)

    def pausar(self, pausado: bool) -> bool:
        """Pausa o reanuda el video. Mientras está en pausa el reloj del análisis se detiene, para que al reanudar
        las reglas no crean que pasó el tiempo (una mochila «quieta» durante la pausa no es un abandono)."""
        if not self.pausable:
            return False
        if pausado and self._pausa_desde is None:
            self._pausa_desde = time.time()
        elif not pausado and self.lector is not None and self.lector.terminado:
            self.lector.rebobinar()                 # el video terminó: ▶ lo vuelve a empezar
        if not pausado and self._pausa_desde is not None:
            self._pausa_total += time.time() - self._pausa_desde
            self._pausa_desde = None
            self._tiempos.clear()                   # que los fps no cuenten el tiempo en pausa
        if self.lector is not None:
            self.lector.pausado = pausado
        self.estado["pausado"] = self._pausa_desde is not None
        if not pausado:
            self.estado["terminado"] = False
        return True

    def _reloj(self) -> float:
        """Hora para el análisis: la del sistema menos el tiempo que el video estuvo en pausa."""
        ahora = self._pausa_desde if self._pausa_desde is not None else time.time()   # en pausa, quieto
        return ahora - self._pausa_total

    def reiniciar(self) -> None:
        """Olvida lo aprendido de la escena (personas, objetos, escenario, falsas alarmas). Lo aplica el propio
        hilo de la cámara en su siguiente cuadro, para no cambiar el estado mientras lo está usando."""
        self._reiniciar = True

    def _aplicar_reinicio(self, conservar_ignorados: bool = False) -> None:
        ignorar = self.reglas.ignorar
        self.hist = Historial()
        self.filtro = FiltroObjetos(abierto=self.filtro.abierto)
        self.reglas = Reglas(self.cfg["reglas"])
        if conservar_ignorados:
            self.reglas.ignorar = ignorar
        if self.detector is not None and hasattr(self.detector, "reiniciar_seguimiento"):
            self.detector.reiniciar_seguimiento()
        self._en_alerta.clear()
        self._tiempos.clear()
        self._reiniciar = False

    def _paso(self):
        if self._reiniciar:
            self._aplicar_reinicio()
        priv = self.cfg["privacidad"]
        if self.demo:
            t = self._reloj() - self._t0
            frame, obs = self.sim.frame(t)
            self.estado["conectada"] = True
            if not priv["analizar_emociones"]:
                for r in obs.rostros:
                    r.probs = None
        else:
            self.estado["conectada"] = self.lector.conectada
            video = self.lector.finito
            if video:
                if self.lector.vuelta != self._vuelta:   # el video volvió a empezar: las personas «saltan» de lugar
                    self._vuelta = self.lector.vuelta
                    self._aplicar_reinicio(conservar_ignorados=True)
                self.estado["terminado"] = self.lector.terminado
                tomado = self.lector.tomar()
                if tomado is None:
                    return                               # el lector todavía no tiene el cuadro siguiente
                frame, t = tomado                        # reloj del análisis = segundo del video
                self.detector.cada_n = int(self.cfg["procesamiento"].get("objetos_cada_n_frames_videos", 2))
            else:
                frame = self.lector.ultimo()
                if frame is None:
                    return
                t = self._reloj()
            ancho_max = self.cfg["procesamiento"]["ancho_max"]
            if frame.shape[1] > ancho_max:
                esc = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, (ancho_max, int(frame.shape[0] * esc)))
            obs = self.detector.procesar(frame, t)
            if self.rostros.disponible and (priv["difuminar_rostros"] or priv["analizar_emociones"]):
                from .modelos import asignar_rostros
                obs.rostros = self.rostros.detectar(frame, priv["analizar_emociones"])
                asignar_rostros(obs.rostros, obs.personas)

        h, w = frame.shape[:2]
        if not self.demo:            # los objetos del modo demo ya vienen limpios
            if not self.lector.finito and len(self._tiempos) > 5:
                # cámara en vivo en una PC lenta: si el detector de objetos pasa pocas veces por segundo, la ventana
                # para juntar 3 detecciones se alarga (con 4 s fijos, a 1 cuadro por segundo nada se confirmaba)
                fps_med = (len(self._tiempos) - 1) / max(1e-3, self._tiempos[-1] - self._tiempos[0])
                cada_n = max(1, self.cfg["modelos"]["objetos_cada_n_frames"])
                self.filtro.ventana = max(4.0, 1.5 * self.filtro.detecciones_min * cada_n / max(fps_med, 0.1))
            obs.objetos = self.filtro.filtrar(obs.objetos, t, self.cfg["reglas"]["objetos"], obs.personas)
        self.hist.actualizar(obs)
        eventos = self.reglas.evaluar(obs, self.hist, w, h, self.zonas)
        ahora = time.time()
        for e in eventos:
            for c in e.cajas:
                for p in obs.personas:
                    if np.allclose(p.caja, c):
                        self._en_alerta[p.id] = ahora
        en_alerta = {pid for pid, ta in self._en_alerta.items() if ahora - ta < 4}

        emociones = {}
        if priv["analizar_emociones"]:
            for p in obs.personas:
                est = self.hist.personas.get(p.id)
                if est and "emo_etiqueta" in est.estado:
                    emociones[p.id] = est.estado["emo_etiqueta"]

        esqueleto = bool(priv.get("solo_esqueleto"))
        if priv["difuminar_rostros"]:
            difuminar(frame, cajas_cabeza(obs))
        if esqueleto:
            solo_esqueleto(frame, obs)
        self.crudo = frame.copy()
        self._tiempos.append(ahora)
        fps = (len(self._tiempos) - 1) / (self._tiempos[-1] - self._tiempos[0]) if len(self._tiempos) > 1 else 0.0
        velocidad = None
        if self.lector is not None and self.lector.finito:
            velocidad = round(fps * self.lector.paso_video, 2)      # segundos de video analizados por segundo real
            enc = (f"{self.nombre}  |  video {t:5.1f} s  |  {fps:.1f} fps"
                   + (f" (camara lenta x{velocidad:.2f})" if velocidad < 0.95 else "") + f"  |  {len(obs.personas)} personas")
        else:
            enc = f"{self.nombre}  |  {time.strftime('%H:%M:%S')}  |  {fps:.1f} fps  |  {len(obs.personas)} personas"
        if self.demo:
            enc += "  |  MODO DEMO (escena simulada)"
        if esqueleto:
            enc += "  |  SOLO ESQUELETO"
        dibujar(frame, obs, self.zonas, en_alerta, emociones, enc, grueso=esqueleto, duenos=self.reglas.duenos(self.hist))
        self.alertas.procesar(self.id, eventos, frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
        self.estado.update(fps=round(fps, 1), personas=len(obs.personas),
                           emociones=dict(Counter(emociones.values())), error=None, ancho=w, alto=h,
                           pausable=self.pausable, pausado=self._pausa_desde is not None, velocidad=velocidad)

    def actualizar_cfg(self, cfg: dict):
        self.cfg = cfg
        self.reglas.actualizar_cfg(cfg["reglas"])

    def detener(self):
        self._parar = True
        if self.lector:
            self.lector.detener()
