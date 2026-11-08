"""Perfiles de rendimiento: cuánto trabajo hace VigIA en cada cuadro.

La precisión del análisis en vivo depende sobre todo de cuántos cuadros por segundo alcanza la computadora:
un tirón de mochila o un golpe duran 1 a 2 s, y con 1-2 cuadros por segundo caen entre dos cuadros. Con
10 cuadros por segundo se ven igual que en un video analizado en cámara lenta. El perfil «rápido» baja el
tamaño de las imágenes que reciben los modelos para llegar a esa cantidad en una laptop sin tarjeta gráfica.

  normal  lo que diga config.yaml
  rapido  personas a 384 px, objetos a 480 px, rostros cada 3 cuadros, imagen de trabajo de 800 px de ancho.
          Pierde algo de alcance (una mochila a más de 5-6 m se ve peor), no de exactitud de cerca, que es
          como se prueba en clase o en una feria.
"""
from __future__ import annotations

PERFILES: dict[str, dict] = {
    "normal": {},
    "rapido": {
        "modelos": {"tamano_imagen": 384, "tamano_imagen_objetos": 480, "objetos_cada_n_frames": 3},
        "procesamiento": {"ancho_max": 800, "rostros_cada_n_frames": 3, "ancho_rostros": 400},
    },
}


def aplicar_perfil(cfg: dict, nombre: str | None) -> str:
    """Aplica el perfil sobre la configuración (in situ) y devuelve el nombre aplicado."""
    nombre = (nombre or "normal").lower().replace("á", "a")
    if nombre not in PERFILES:
        raise ValueError(f"Perfil desconocido «{nombre}»: use {', '.join(PERFILES)}")
    for seccion, valores in PERFILES[nombre].items():
        cfg.setdefault(seccion, {}).update(valores)
    cfg.setdefault("procesamiento", {})["perfil"] = nombre
    return nombre
