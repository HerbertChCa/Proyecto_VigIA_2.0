"""Aviso de videovigilancia para imprimir y colocar en el lugar (página /aviso).

Responde a la PE3 del artículo: la aceptación es condicional y depende de la finalidad declarada,
del tipo de función y de la confianza en quien gestiona los datos (E45-E48). El aviso hace explícitas
esas tres cosas y lo que el sistema NO hace. En Perú, la Ley N.° 29733 y la directiva de la ANPD sobre
videovigilancia exigen informar a las personas grabadas.
"""
from __future__ import annotations

from html import escape

DESCRIPCION = {
    "violencia": "posibles peleas o agresiones",
    "caida": "caídas y personas que quedan en el suelo",
    "merodeo": "permanencia prolongada en zonas definidas",
    "intrusion": "ingreso a zonas restringidas",
    "objetos": "objetos abandonados y su posible sustracción",
}


def pagina_aviso(cfg: dict, n_camaras: int) -> str:
    p = cfg["privacidad"]
    r = cfg["reglas"]
    activas = [d for k, d in DESCRIPCION.items() if r.get(k, {}).get("activa")]
    if p.get("analizar_emociones"):
        activas.append("expresiones faciales, solo con fines informativos")
    no_hace = ["No identifica a las personas: no usa reconocimiento facial ni guarda identidades."]
    if p.get("difuminar_rostros"):
        no_hace.append("Los rostros se pixelan antes de mostrarse o guardarse.")
    no_hace += [
        "No graba video de forma continua: solo guarda una imagen cuando se genera una alerta.",
        "Procesa las imágenes en un equipo local; no las envía a internet.",
        "No toma decisiones por sí mismo: cada alerta la revisa una persona.",
    ]
    li = lambda xs: "".join(f"<li>{escape(x)}</li>" for x in xs)
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Aviso de videovigilancia</title>
<style>
 body{{font:16px/1.5 "Segoe UI",system-ui,sans-serif;color:#111;background:#fff;margin:0}}
 main{{max-width:760px;margin:24px auto;padding:28px 32px;border:3px solid #111;border-radius:14px}}
 h1{{font-size:28px;margin:0 0 4px}} .sub{{color:#444;margin:0 0 18px}}
 h2{{font-size:17px;margin:18px 0 6px;text-transform:uppercase;letter-spacing:.5px}}
 ul{{margin:0;padding-left:22px}} .pie{{font-size:13px;color:#444;margin-top:20px}}
 .ojo{{float:right;font-size:48px;line-height:1}}
 @media print{{main{{margin:0;border-width:4px}} .noimprimir{{display:none}}}}
</style></head><body><main>
<div class="ojo" aria-hidden="true">◉</div>
<h1>Zona videovigilada con análisis automático</h1>
<p class="sub">{n_camaras} cámara(s) con detección automática de situaciones de riesgo.</p>
<h2>Finalidad</h2><p>{escape(str(p.get("finalidad", "")))}</p>
<h2>Qué detecta</h2><ul>{li(activas)}</ul>
<h2>Qué no hace</h2><ul>{li(no_hace)}</ul>
<h2>Conservación</h2><p>Las imágenes de las alertas se eliminan a los {int(p.get("dias_retencion_capturas", 7))} días.</p>
<h2>Responsable y consultas</h2><p>{escape(str(p.get("responsable", "")))}<br>{escape(str(p.get("contacto", "")))}</p>
<p class="pie">Puede ejercer sus derechos de acceso, rectificación, cancelación y oposición conforme a la
Ley N.° 29733, Ley de Protección de Datos Personales.</p>
<p class="pie noimprimir">Use Ctrl + P para imprimir este aviso.</p>
</main></body></html>"""
