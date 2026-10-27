"""Servidor web del panel de VigIA (Starlette + uvicorn)."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .alertas import GestorAlertas
from .aviso import pagina_aviso
from .camara import Trabajador

log = logging.getLogger("vigia.servidor")
RAIZ = Path(__file__).resolve().parent.parent
WEB = RAIZ / "web"


def _mezclar(base: dict, cambios: dict) -> dict:
    for k, v in cambios.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _mezclar(base[k], v)
        else:
            base[k] = v
    return base


class Sistema:
    def __init__(self, cfg: dict, datos: Path, demo: bool):
        self.datos = datos
        datos.mkdir(parents=True, exist_ok=True)
        self.demo = demo
        self.cfg = copy.deepcopy(cfg)
        self._f_ajustes = datos / "ajustes.json"
        self._f_zonas = datos / "zonas.json"
        self._f_camaras = datos / "camaras.json"
        if self._f_ajustes.exists():
            _mezclar(self.cfg, json.loads(self._f_ajustes.read_text(encoding="utf-8")))
        self.zonas: dict[str, list] = json.loads(self._f_zonas.read_text(encoding="utf-8")) if self._f_zonas.exists() else {}
        self.alertas = GestorAlertas(datos, self.cfg["alertas"]["enfriamiento"])
        self.alertas.limpiar_antiguas(self.cfg["privacidad"]["dias_retencion_capturas"])
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
        if self._f_camaras.exists():
            extra = [c for c in json.loads(self._f_camaras.read_text(encoding="utf-8")) if c["id"] != cid]
            self._f_camaras.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")

    def guardar_zonas(self, cid: str, zonas: list) -> None:
        lista = self.zonas.setdefault(cid, [])
        lista[:] = zonas          # se modifica la misma lista que usa el trabajador
        self._f_zonas.write_text(json.dumps(self.zonas, ensure_ascii=False, indent=2), encoding="utf-8")

    def ajustes(self) -> dict:
        return {"privacidad": self.cfg["privacidad"], "reglas": self.cfg["reglas"]}

    def aplicar_ajustes(self, cambios: dict) -> None:
        cambios = {k: v for k, v in cambios.items() if k in ("privacidad", "reglas")}
        _mezclar(self.cfg, cambios)
        previo = json.loads(self._f_ajustes.read_text(encoding="utf-8")) if self._f_ajustes.exists() else {}
        self._f_ajustes.write_text(json.dumps(_mezclar(previo, cambios), ensure_ascii=False, indent=2), encoding="utf-8")
        for t in self.trabajadores.values():
            t.actualizar_cfg(self.cfg)

    def estado(self) -> dict:
        return {"camaras": [t.estado for t in self.trabajadores.values()], "resumen": self.alertas.resumen(),
                "operador": self.alertas.desempeno_operador(), "demo": self.demo}


def crear_app(cfg: dict, datos: Path, demo: bool = False) -> Starlette:
    sis: Sistema | None = None

    async def inicio():
        nonlocal sis
        sis = Sistema(cfg, datos, demo)
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
                if len(pts) >= 3 and z.get("tipo") in ("restringida", "merodeo"):
                    limpias.append({"id": z.get("id") or uuid.uuid4().hex[:6], "nombre": str(z.get("nombre", "Zona"))[:40],
                                    "tipo": z["tipo"], "puntos": pts})
            sis.guardar_zonas(cid, limpias)
        return JSONResponse(sis.zonas.get(cid, []))

    async def camaras(req: Request):
        if req.method == "POST":
            c = await req.json()
            fuente = str(c.get("fuente", "")).strip()
            if not fuente:
                return JSONResponse({"error": "Indica la fuente"}, status_code=400)
            cam = {"id": "cam" + uuid.uuid4().hex[:5], "nombre": str(c.get("nombre") or fuente)[:40], "fuente": fuente}
            sis.agregar_camara(cam)
            return JSONResponse(cam)
        return JSONResponse([t.estado for t in sis.trabajadores.values()])

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
        Route("/api/ajustes", ajustes, methods=["GET", "PUT"]),
        Mount("/static", StaticFiles(directory=WEB / "static"), name="static"),
    ]
    @asynccontextmanager
    async def vida(_app):
        await inicio()
        yield
        for t in sis.trabajadores.values():
            t.detener()

    app = Starlette(routes=rutas, lifespan=vida)
    return app
