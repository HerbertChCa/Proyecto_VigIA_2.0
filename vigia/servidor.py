"""Servidor web del panel de VigIA (Starlette + uvicorn)."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import socket
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .alertas import GestorAlertas
from .aviso import pagina_aviso
from .busqueda import Buscador
from .camara import Trabajador
from .descargas import PAGINAS_DE_VIDEO, Descargas
from .descargas import disponible as descargas_disponibles
from .encuesta import LIKERT, PREGUNTAS, Encuesta
from .reto import RETOS, TIEMPOS_FERIA, Marcador

TIPOS_ZONA = ("restringida", "merodeo", "protegida")
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
MAX_SUBIDA = 2 * 1024 ** 3                       # 2 GB por video
HORA = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")   # HH:MM de 00:00 a 23:59

log = logging.getLogger("vigia.servidor")
RAIZ = Path(__file__).resolve().parent.parent
WEB = RAIZ / "web"


def ip_local() -> str:
    """IP de este equipo en la red local (la que deben usar los celulares para abrir la encuesta)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))       # no envía nada: solo elige la interfaz de salida
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# En modo feria el servidor escucha en la red local para que los celulares abran la encuesta.
# Desde otros equipos solo se permiten estas páginas; el panel, los ajustes y las alertas quedan
# accesibles únicamente desde la PC de VigIA.
PUBLICAS = {("GET", "/encuesta"), ("GET", "/encuesta/resultados"), ("GET", "/api/encuesta/preguntas"),
            ("POST", "/api/encuesta"), ("GET", "/api/encuesta/resultados"), ("GET", "/aviso")}


class SoloLocal:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            cliente = (scope.get("client") or ("",))[0]
            ruta, metodo = scope["path"], scope["method"]
            publica = (metodo, ruta) in PUBLICAS or (metodo == "GET" and ruta.startswith("/static/"))
            if cliente not in ("127.0.0.1", "::1") and not publica:
                await JSONResponse({"error": "Disponible solo en el equipo de VigIA."}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _mezclar(base: dict, cambios: dict) -> dict:
    for k, v in cambios.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _mezclar(base[k], v)
        else:
            base[k] = v
    return base


class Sistema:
    def __init__(self, cfg: dict, datos: Path, demo: bool, feria: bool = False):
        self.datos = datos
        datos.mkdir(parents=True, exist_ok=True)
        self.demo = demo
        self.feria = feria
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("busqueda", {"activa": False, "modelo": "xlm-roberta-base-ViT-B-32",
                                         "pesos": "laion5b_s13b_b90k"})
        self._f_ajustes = datos / "ajustes.json"
        self._f_zonas = datos / "zonas.json"
        self._f_camaras = datos / "camaras.json"
        if self._f_ajustes.exists():
            _mezclar(self.cfg, json.loads(self._f_ajustes.read_text(encoding="utf-8")))
        self.zonas: dict[str, list] = json.loads(self._f_zonas.read_text(encoding="utf-8")) if self._f_zonas.exists() else {}
        if feria:                                 # retos de segundos, no de minutos
            for ruta, valor in TIEMPOS_FERIA.items():
                regla, clave = ruta.split(".")
                self.cfg["reglas"][regla][clave] = valor
            self.cfg["alertas"]["enfriamiento"] = 5
        self.alertas = GestorAlertas(datos, self.cfg["alertas"]["enfriamiento"])
        self.alertas.limpiar_antiguas(self.cfg["privacidad"]["dias_retencion_capturas"])
        self.descargas = Descargas(datos / "videos")
        self.encuesta = Encuesta(datos)
        self.marcador = Marcador(datos / "reto.json")
        self.buscador = Buscador(datos / "capturas", datos / "indice_clip.npz")
        self.alertas.al_guardar = self.buscador.agregar
        self._iniciar_busqueda()
        camaras = list(self.cfg["camaras"])
        if self._f_camaras.exists():
            camaras += json.loads(self._f_camaras.read_text(encoding="utf-8"))
        if demo:
            from .simulador import ZONAS_DEMO
            camaras = [{"id": "demo", "nombre": "Escena demo", "fuente": "demo"}]
            self.zonas.setdefault("demo", copy.deepcopy(ZONAS_DEMO))
            # tiempos cortos para que cada comportamiento aparezca en un recorrido de 100 s
            self.cfg["reglas"]["merodeo"]["tiempo_max"] = 12
            self.cfg["reglas"]["objetos"]["tiempo_abandono"] = 10
            self.cfg["privacidad"]["analizar_emociones"] = True   # la escena simulada muestra las expresiones
        self.trabajadores: dict[str, Trabajador] = {}
        for cam in camaras:
            self.agregar_camara(cam, guardar=False)

    def agregar_camara(self, cam: dict, guardar: bool = True) -> None:
        self.zonas.setdefault(cam["id"], [])
        self.trabajadores[cam["id"]] = Trabajador(cam, self.cfg, self.alertas, self.zonas[cam["id"]],
                                                  demo=str(cam["fuente"]) == "demo")
        if guardar:
            extra = json.loads(self._f_camaras.read_text(encoding="utf-8")) if self._f_camaras.exists() else []
            extra.append(cam)
            self._f_camaras.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")

    def quitar_camara(self, cid: str) -> None:
        t = self.trabajadores.pop(cid, None)
        if t:
            t.detener()
            self._borrar_si_subido(t.fuente)
        if self._f_camaras.exists():
            extra = [c for c in json.loads(self._f_camaras.read_text(encoding="utf-8")) if c["id"] != cid]
            self._f_camaras.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")

    def _borrar_si_subido(self, fuente) -> None:
        """Un video subido desde el panel se borra al quitar su cámara (no se conservan grabaciones)."""
        ruta = Path(str(fuente))
        if ruta.parent.resolve() != (self.datos / "videos").resolve():
            return

        def borrar(intentos=5):
            try:
                ruta.unlink(missing_ok=True)
            except OSError:                     # en Windows el lector puede tenerlo abierto un instante más
                if intentos:
                    threading.Timer(2.0, borrar, (intentos - 1,)).start()
        threading.Timer(1.0, borrar).start()

    def guardar_zonas(self, cid: str, zonas: list) -> None:
        lista = self.zonas.setdefault(cid, [])
        lista[:] = zonas          # se modifica la misma lista que usa el trabajador
        self._f_zonas.write_text(json.dumps(self.zonas, ensure_ascii=False, indent=2), encoding="utf-8")

    def _iniciar_busqueda(self) -> None:
        b = self.cfg["busqueda"]
        if b.get("activa") and self.buscador.estado == "apagada":
            self.buscador.iniciar(b["modelo"], b["pesos"])

    def ajustes(self) -> dict:
        return {"privacidad": self.cfg["privacidad"], "reglas": self.cfg["reglas"], "busqueda": self.cfg["busqueda"]}

    def aplicar_ajustes(self, cambios: dict) -> None:
        cambios = {k: v for k, v in cambios.items() if k in ("privacidad", "reglas", "busqueda")}
        _mezclar(self.cfg, cambios)
        previo = json.loads(self._f_ajustes.read_text(encoding="utf-8")) if self._f_ajustes.exists() else {}
        self._f_ajustes.write_text(json.dumps(_mezclar(previo, cambios), ensure_ascii=False, indent=2), encoding="utf-8")
        for t in self.trabajadores.values():
            t.actualizar_cfg(self.cfg)
        self._iniciar_busqueda()

    def estado(self) -> dict:
        return {"camaras": [t.estado for t in self.trabajadores.values()], "resumen": self.alertas.resumen(),
                "operador": self.alertas.desempeno_operador(), "demo": self.demo, "feria": self.feria,
                "solo_esqueleto": bool(self.cfg["privacidad"].get("solo_esqueleto"))}


def crear_app(cfg: dict, datos: Path, demo: bool = False, feria: bool = False) -> Starlette:
    sis: Sistema | None = None

    async def inicio():
        nonlocal sis
        sis = Sistema(cfg, datos, demo, feria)
        app.state.sistema = sis

    async def index(_):
        return FileResponse(WEB / "index.html")

    async def video(req: Request):
        cid = req.path_params["cid"]

        async def flujo():
            ultimo = None
            while True:
                if await req.is_disconnected():
                    break
                t = sis.trabajadores.get(cid)
                if t is None:
                    break
                if t.jpeg is not None and t.jpeg is not ultimo:
                    ultimo = t.jpeg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                           str(len(ultimo)).encode() + b"\r\n\r\n" + ultimo + b"\r\n")
                await asyncio.sleep(0.05)

        return StreamingResponse(flujo(), media_type="multipart/x-mixed-replace; boundary=frame")

    async def captura(req: Request):
        t = sis.trabajadores.get(req.path_params["cid"])
        if t is None or t.jpeg is None:
            return Response(status_code=404)
        datos_img = t.jpeg
        if req.query_params.get("crudo") and t.crudo is not None:   # para el editor de zonas
            ok, buf = cv2.imencode(".jpg", t.crudo, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                datos_img = buf.tobytes()
        return Response(datos_img, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def eventos(req: Request):
        q = sis.alertas.suscribir()

        async def flujo():
            try:
                yield GestorAlertas.sse({"evento": "estado", "datos": sis.estado()})
                while not await req.is_disconnected():
                    try:
                        datos = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        datos = {"evento": "estado", "datos": sis.estado()}
                    yield GestorAlertas.sse(datos)
            finally:
                sis.alertas.desuscribir(q)

        return StreamingResponse(flujo(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def estado(_):
        return JSONResponse(sis.estado())

    async def alertas(req: Request):
        lim = min(int(req.query_params.get("limite", 100)), 500)
        return JSONResponse(sis.alertas.listar(lim, req.query_params.get("tipo") or None))

    async def atender(req: Request):
        sis.alertas.atender(int(req.path_params["aid"]))
        return JSONResponse({"ok": True})

    async def veredicto(req: Request):
        valor = (await req.json()).get("valor")
        try:
            sis.alertas.dar_veredicto(int(req.path_params["aid"]), valor)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "operador": sis.alertas.desempeno_operador()})

    async def alertas_csv(_):
        return Response(sis.alertas.exportar_csv(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=alertas_vigia.csv"})

    async def aviso(_):
        return HTMLResponse(pagina_aviso(sis.cfg, len(sis.trabajadores)))

    async def archivo_captura(req: Request):
        nombre = req.path_params["nombre"]
        if not re.fullmatch(r"[\w\-.]+\.jpg", nombre):
            return Response(status_code=400)
        ruta = datos / "capturas" / nombre
        return FileResponse(ruta) if ruta.exists() else Response(status_code=404)

    async def zonas(req: Request):
        cid = req.path_params["cid"]
        if cid not in sis.trabajadores:
            return JSONResponse({"error": "cámara desconocida"}, status_code=404)
        if req.method == "PUT":
            cuerpo = await req.json()
            limpias = []
            for z in cuerpo:
                pts = [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in z["puntos"]]
                if len(pts) >= 3 and z.get("tipo") in TIPOS_ZONA:
                    zona = {"id": z.get("id") or uuid.uuid4().hex[:6], "nombre": str(z.get("nombre", "Zona"))[:40],
                            "tipo": z["tipo"], "puntos": pts}
                    h = z.get("horario") or {}
                    if HORA.match(str(h.get("desde", ""))) and HORA.match(str(h.get("hasta", ""))) \
                            and h["desde"] != h["hasta"]:
                        zona["horario"] = {"desde": h["desde"], "hasta": h["hasta"]}
                    limpias.append(zona)
            sis.guardar_zonas(cid, limpias)
        return JSONResponse(sis.zonas.get(cid, []))

    async def camaras(req: Request):
        if req.method == "POST":
            c = await req.json()
            fuente = str(c.get("fuente", "")).strip()
            if not fuente:
                return JSONResponse({"error": "Indica la fuente"}, status_code=400)
            if PAGINAS_DE_VIDEO.search(fuente):
                # una página (YouTube, Facebook, TikTok…) no es un video: se descarga con yt-dlp y se agrega al terminar
                if not descargas_disponibles():
                    return JSONResponse({"error": "Para usar enlaces de YouTube, Facebook o TikTok instale yt-dlp "
                                         "(pip install -r requirements-enlaces.txt) y, para YouTube, Deno "
                                         "(winget install DenoLand.Deno). Luego reinicie VigIA. También puede "
                                         "guardar el video y subirlo en «Video de mi PC»."}, status_code=400)
                titulo = str(c.get("nombre") or "").strip()
                titulo = "" if titulo == fuente else titulo[:40]

                def al_terminar(ruta: str, nombre: str) -> dict:
                    cam = {"id": "cam" + uuid.uuid4().hex[:5], "nombre": nombre, "fuente": ruta}
                    sis.agregar_camara(cam)
                    return cam
                return JSONResponse({"descarga": sis.descargas.iniciar(fuente, titulo, al_terminar)}, status_code=202)
            cam = {"id": "cam" + uuid.uuid4().hex[:5], "nombre": str(c.get("nombre") or fuente)[:40], "fuente": fuente}
            sis.agregar_camara(cam)
            return JSONResponse(cam)
        return JSONResponse([t.estado for t in sis.trabajadores.values()])

    async def subir_video(req: Request):
        """Recibe un video desde el navegador (el cuerpo es el archivo) y lo agrega como cámara."""
        original = Path(req.query_params.get("nombre", "video.mp4")).name
        ext = Path(original).suffix.lower()
        if ext not in EXT_VIDEO:
            return JSONResponse({"error": f"Formato no admitido ({ext or 'sin extensión'}). Use {', '.join(EXT_VIDEO)}."},
                                status_code=400)
        carpeta = sis.datos / "videos"
        carpeta.mkdir(exist_ok=True)
        base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original).stem)[:40] or "video"
        destino = carpeta / f"{datetime.now():%Y%m%d_%H%M%S}_{base}{ext}"
        total = 0
        with open(destino, "wb") as f:
            async for trozo in req.stream():
                total += len(trozo)
                if total > MAX_SUBIDA:
                    break
                f.write(trozo)
        if total > MAX_SUBIDA or total == 0:
            destino.unlink(missing_ok=True)
            return JSONResponse({"error": "El video está vacío o supera los 2 GB."}, status_code=413 if total else 400)
        cap = cv2.VideoCapture(str(destino))      # se comprueba que realmente sea un video legible
        ok = cap.read()[0]
        cap.release()
        if not ok:
            destino.unlink(missing_ok=True)
            return JSONResponse({"error": "No se pudo leer el archivo como video."}, status_code=400)
        titulo = (req.query_params.get("titulo") or Path(original).stem).strip()[:40]
        cam = {"id": "cam" + uuid.uuid4().hex[:5], "nombre": titulo, "fuente": str(destino)}
        sis.agregar_camara(cam)
        return JSONResponse(cam)

    # ---- feria: retos
    def pagina(archivo: str):
        async def servir(_):
            return FileResponse(WEB / archivo)
        return servir

    def zonas_por_camara() -> dict:
        return {cid: sorted({z.get("tipo") for z in sis.zonas.get(cid, [])}) for cid in sis.trabajadores}

    async def reto(_):
        return JSONResponse({"retos": RETOS, "marcador": sis.marcador.resumen(), "feria": sis.feria,
                             "zonas": zonas_por_camara(),
                             "emociones": bool(sis.cfg["privacidad"].get("analizar_emociones"))})

    async def reto_resultado(req: Request):
        d = await req.json()
        seg = d.get("segundos")
        try:
            resumen = sis.marcador.registrar(str(d.get("reto")), bool(d.get("detectado")),
                                             float(seg) if seg is not None else None)
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(resumen)

    # ---- feria: encuesta
    def url_encuesta() -> str:
        return f"http://{ip_local()}:{sis.cfg['servidor']['puerto']}/encuesta"

    async def encuesta_preguntas(_):
        return JSONResponse({"preguntas": PREGUNTAS, "likert": LIKERT})

    async def encuesta_responder(req: Request):
        try:
            sis.encuesta.guardar(await req.json())
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True})

    async def encuesta_resultados(_):
        return JSONResponse({**sis.encuesta.resultados(), "url": url_encuesta(), "feria": sis.feria})

    async def encuesta_csv(_):
        return Response(sis.encuesta.exportar_csv(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=encuesta_vigia.csv"})

    async def encuesta_qr(_):
        qr = cv2.QRCodeEncoder.create().encode(url_encuesta())
        qr = cv2.resize(qr, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        qr = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        ok, buf = cv2.imencode(".png", qr)
        return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    # ---- búsqueda con CLIP
    async def buscar(req: Request):
        b = sis.buscador
        info = {"estado": b.estado, "total": b.total, "pendientes": b.pendientes()}
        frase = req.query_params.get("q", "")
        if not frase:
            return JSONResponse({**info, "resultados": []})
        try:
            k = min(max(int(req.query_params.get("k", 12)), 1), 48)
            res = await asyncio.to_thread(b.buscar, frase, k)
        except RuntimeError as e:
            return JSONResponse({**info, "error": str(e)}, status_code=409)
        alertas = sis.alertas.por_capturas([r["captura"] for r in res])
        for r in res:
            r["alerta"] = alertas.get(r["captura"])
        return JSONResponse({**info, "resultados": res})

    async def descarga(req: Request):
        d = sis.descargas.estado(req.path_params["tid"])
        return JSONResponse(d) if d else JSONResponse({"error": "descarga desconocida"}, status_code=404)

    async def quitar(req: Request):
        sis.quitar_camara(req.path_params["cid"])
        return JSONResponse({"ok": True})

    async def ajustes(req: Request):
        if req.method == "PUT":
            sis.aplicar_ajustes(await req.json())
        return JSONResponse(sis.ajustes())

    rutas = [
        Route("/", index),
        Route("/video/{cid}", video),
        Route("/api/captura/{cid}", captura),
        Route("/api/eventos", eventos),
        Route("/api/estado", estado),
        Route("/api/alertas", alertas),
        Route("/api/alertas/{aid:int}/atender", atender, methods=["POST"]),
        Route("/api/alertas/{aid:int}/veredicto", veredicto, methods=["POST"]),
        Route("/api/alertas.csv", alertas_csv),
        Route("/aviso", aviso),
        Route("/capturas/{nombre}", archivo_captura),
        Route("/api/camaras/{cid}/zonas", zonas, methods=["GET", "PUT"]),
        Route("/api/camaras", camaras, methods=["GET", "POST"]),
        Route("/api/camaras/{cid}", quitar, methods=["DELETE"]),
        Route("/api/videos", subir_video, methods=["PUT"]),
        Route("/api/descargas/{tid}", descarga),
        Route("/api/ajustes", ajustes, methods=["GET", "PUT"]),
        Route("/reto", pagina("reto.html")),
        Route("/api/reto", reto),
        Route("/api/reto/resultado", reto_resultado, methods=["POST"]),
        Route("/encuesta", pagina("encuesta.html")),
        Route("/encuesta/resultados", pagina("resultados.html")),
        Route("/api/encuesta/preguntas", encuesta_preguntas),
        Route("/api/encuesta", encuesta_responder, methods=["POST"]),
        Route("/api/encuesta/resultados", encuesta_resultados),
        Route("/api/encuesta.csv", encuesta_csv),
        Route("/api/encuesta/qr.png", encuesta_qr),
        Route("/api/buscar", buscar),
        Mount("/static", StaticFiles(directory=WEB / "static"), name="static"),
    ]
    @asynccontextmanager
    async def vida(_app):
        await inicio()
        yield
        for t in sis.trabajadores.values():
            t.detener()
        sis.buscador.detener()

    app = Starlette(routes=rutas, lifespan=vida)
    return SoloLocal(app) if feria else app
