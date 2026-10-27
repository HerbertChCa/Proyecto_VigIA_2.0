# VigIA: reconocimiento de comportamiento humano en cámaras de seguridad

Prototipo de software derivado de la revisión sistemática *Reconocimiento del comportamiento humano en cámaras
de seguridad mediante visión artificial y aprendizaje profundo (2021-2026)*. Analiza en tiempo real la webcam o
cámaras IP (RTSP), detecta comportamientos de riesgo y los muestra en un panel web con evidencia y explicación.

| Comportamiento | Cómo se detecta | Alerta |
|---|---|---|
| Violencia o pelea | Dos personas a menos de 1.2 alturas con muñecas, codos o tobillos moviéndose rápido respecto del torso (≥ 1.8 alturas/s durante ≥ 0.6 s) | Alta |
| Caída | El centro de la cadera baja ≥ 0.35 alturas en menos de 2.5 s y el tronco queda a más de 60° de la vertical | Alta |
| Persona en el suelo | Tronco horizontal durante ≥ 5 s | Alta |
| Intrusión | Los pies de una persona entran en una zona marcada como «restringida» | Alta |
| Merodeo | Permanencia mayor a 60 s en una zona de «merodeo» (o en la cámara si no hay zonas) | Media |
| Objeto abandonado | Mochila, bolso o maleta inmóvil ≥ 30 s sin nadie a menos de 2.5 alturas | Media |
| Posible hurto | Un objeto que estaba quieto y asociado a una persona se desplaza junto a otra persona | Alta |
| Expresión facial (opcional) | Alegría, neutral, sorpresa, tristeza, enojo, miedo o desagrado, por persona; apagada por defecto | Informativa |

Todas las distancias se miden en «alturas de persona», así que las reglas no dependen de la resolución ni de la
distancia a la cámara. Todos los umbrales se cambian en `config.yaml` o desde **Ajustes** en el panel.

## Fundamento en el artículo de revisión

VigIA es el producto del artículo *Reconocimiento del comportamiento humano en cámaras de seguridad mediante visión
artificial y aprendizaje profundo: revisión sistemática de desafíos técnicos, percepción social y tendencias (2021-2026)*,
que analizó 48 estudios (E01-E48). Cada decisión de diseño responde a un hallazgo o a una brecha de su agenda (Tabla 11):

| Hallazgo del artículo | Decisión en VigIA | Dónde |
|---|---|---|
| PE2: dominan las CNN 2D supervisadas (47.7 %); los enfoques por esqueleto no conservan la apariencia y podrían mejorar la aceptación (E18, E39; sección 4.2) | Analizar la **pose** (17 puntos) con reglas, no la apariencia | `modelos.py`, `reglas.py` |
| El corpus estudia violencia (RWF-2000: E18, E20, E24, E30, E39), agresión y caídas con reglas en el borde (E24), merodeo con YOLO + seguimiento (E22) y robo o hurto (E17, E37) | Siete comportamientos con esas mismas técnicas: YOLO11 + ByteTrack + reglas | `reglas.py` |
| PE4: el tiempo real preocupa a 36 de 44 estudios, pero solo 17 miden la velocidad; la salida es el borde y la eficiencia (E24, E34, E36, E40, E44) | Modelos *nano* en CPU, detector de objetos 1 de cada 3 frames, FPS visible en el panel y latencia medida | `modelos.py`, `camara.py` |
| Brecha metodológica: 15 de 44 reportan solo *accuracy*; falta un conjunto mínimo de métricas (AUC, F1, latencia y hardware) | La evaluación entrega esa **ficha** completa en `ficha_evaluacion.md` | `scripts/evaluar_rwf2000.py` |
| Brecha de validación: solo 2 de 44 evalúan con datos externos y el desempeño cae 4 a 10 puntos (E20, E37) | Modo `--umbral` fijo para validar en otros datos y **veredicto del operador** que mide la precisión real de cada instalación | `evaluar_rwf2000.py`, `alertas.py` |
| Brecha contextual: un solo estudio latinoamericano (E12) y solo el 9.1 % graba material propio | Funciona con cámaras RTSP reales y evalúa videos propios con la estructura `Fight/` y `NonFight/` | `camara.py`, `evaluar_rwf2000.py` |
| Brecha de explicabilidad: solo 3 estudios usan Grad-CAM (E25, E37, E41) y ninguno lo evalúa con operadores | Cada alerta trae su medida, su captura y los botones **Confirmar / Falsa alarma**, que permiten medir si la explicación ayuda | `reglas.py`, `alertas.py`, `web/` |
| Solo 6 de 44 tratan la privacidad | Rostros pixelados por defecto, procesamiento local, sin grabación continua, capturas borradas a los 7 días | `dibujo.py`, `config.yaml` |
| PE3: la aceptación es condicional al propósito (82.2 % para un fin protector, E45), el reconocimiento facial es poco aceptado (36 %, E46) y la confianza depende de la institución (E48) | Página **/aviso** con finalidad, responsable y lo que el sistema no hace; sin reconocimiento facial; emociones opcionales y apagadas por defecto | `aviso.py`, `config.yaml` |
| Brecha tecnológica: 0 de 44 usan modelos de visión-lenguaje | No se incluyen; la ficha de evaluación permite compararlos después con el mismo protocolo | Trabajo futuro |

Las implicancias prácticas del artículo piden exigir AUC o F1 en cámaras propias, la latencia medida en el hardware
que se usará y salvaguardas comunicadas a la población. VigIA entrega las tres: la ficha de evaluación, el veredicto
del operador y el aviso de videovigilancia.

## Requisitos

- Windows 10/11, Linux o macOS, con **Python 3.10, 3.11 o 3.12** (python.org; marque «Add python.exe to PATH»). Python 3.13 o
  más nuevo puede no tener PyTorch disponible. Si ya tiene otra versión, instale la 3.12 al lado: `winget install Python.Python.3.12`;
  `iniciar.bat` la elige solo. El modo demo funciona con cualquier Python 3.10 o superior, porque no instala YOLO.
- 8 GB de RAM. Funciona solo con CPU; una GPU NVIDIA lo acelera, pero no es necesaria.
- Conexión a internet solo la primera vez (dependencias y modelos, unos 600 MB en total).

## Instalación y uso en Windows

1. Descomprima la carpeta `vigia`.
2. **Modo demo** (sin cámara ni modelos): doble clic en `demo.bat`. Se abre el panel con una escena simulada de
   100 s donde aparecen todos los comportamientos. Sirve para probar y para exponer.
3. **Con la webcam:** doble clic en `iniciar.bat`. La primera vez crea un entorno de Python, instala las dependencias
   y descarga los modelos; luego abre `http://127.0.0.1:8000`.
4. **Cámara IP:** en el panel, **+ Cámara** y escriba la dirección RTSP. Ejemplos:
   - Hikvision: `rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101`
   - Dahua: `rtsp://usuario:clave@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
   - Un video: la ruta a un `.mp4`, que se reproduce en bucle (útil para probar con clips de UCF-Crime o RWF-2000).
5. **Zonas:** botón **Zonas**, elija la cámara, haga clic sobre la imagen para marcar los vértices y doble clic para
   cerrar el polígono. Las «restringidas» generan intrusión; las de «merodeo», alerta por permanencia.

En Linux o macOS: `./iniciar.sh` o `./iniciar.sh --demo`.

Comandos manuales, si prefiere no usar los `.bat`:

```bash
python -m venv .venv && .venv\Scripts\activate      # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/descargar_modelos.py
python -m vigia            # o: python -m vigia --demo
```

## Evaluación con RWF-2000

RWF-2000 tiene 2 000 clips de 5 s de cámaras de seguridad reales, con y sin pelea. Se solicita a sus autores en
[github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).

```bash
python scripts/evaluar_rwf2000.py --datos D:/RWF-2000/val --max-videos 100
```

El script calcula la pose de cada clip una vez, prueba varios umbrales y muestra accuracy, precisión, recall, F1,
AUC-ROC, la latencia por frame, la resolución y el hardware. Deja el detalle por video en `resultados_rwf2000.csv` y
una ficha lista para el informe en `ficha_evaluacion.md`. Use el mejor umbral en `config.yaml`
(`reglas.violencia.velocidad_extremidades`). El umbral por defecto (1.8) es un valor de partida, no un resultado medido.

Para la validación externa que pide el artículo, calibre en un conjunto y evalúe con el umbral fijo en otro, por ejemplo
videos propios grabados en Arequipa con las carpetas `Fight/` y `NonFight/`:

```bash
python scripts/evaluar_rwf2000.py --datos D:/RWF-2000/train --max-videos 200
python scripts/evaluar_rwf2000.py --datos D:/videos_arequipa --umbral 1.8
```

En uso real, los botones **Confirmar** y **Falsa alarma** de cada alerta miden la precisión en la instalación; el panel la
muestra y **Exportar CSV** descarga todas las alertas con su veredicto.

## Pruebas

```bash
python -m unittest discover pruebas -v
```

Comprueban la geometría (ángulo del tronco, polígonos), que cada comportamiento del guion demo se detecte una sola vez
y en el momento correcto, que dos personas que se cruzan caminando no se marquen como pelea, que una persona tendida
desde el inicio no cuente como caída, que el gestor de alertas evite duplicados, guarde la evidencia y calcule la
precisión con los veredictos, que el AUC del script de evaluación sea correcto y que el aviso declare la finalidad.

## Estructura

```
vigia/
├── iniciar.bat, demo.bat, iniciar.sh   inicio con un clic
├── config.yaml                         cámaras, modelos, umbrales y privacidad
├── vigia/
│   ├── modelos.py      YOLO11 pose + seguimiento ByteTrack, objetos, rostros (YuNet) y expresión (MobileFaceNet)
│   ├── historial.py    memoria temporal de cada persona y objeto seguido
│   ├── reglas.py       violencia, caída, merodeo, intrusión, abandono, hurto y emociones, con su explicación
│   ├── camara.py       lectura de webcam/RTSP/archivo con reconexión y análisis en un hilo por cámara
│   ├── alertas.py      enfriamiento, base SQLite, capturas, veredicto del operador, CSV y envío al panel (SSE)
│   ├── aviso.py        aviso de videovigilancia para imprimir (finalidad, qué detecta, qué no hace)
│   ├── dibujo.py       anotaciones y pixelado de rostros
│   ├── simulador.py    escena sintética del modo demo
│   └── servidor.py     API y video MJPEG (Starlette)
├── web/                panel (HTML, CSS y JavaScript sin dependencias)
├── scripts/            descarga de modelos y evaluación con RWF-2000
├── pruebas/            pruebas automáticas
└── datos/              alertas.db, capturas y ajustes (se crea al ejecutar)
```

## Privacidad, ética y marco legal

- Instale cámaras con fines definidos y avisos visibles: complete `finalidad`, `responsable` y `contacto` en `config.yaml`
  e imprima la página **Aviso** del panel. En Perú rigen la Ley N.° 29733 de Protección de Datos
  Personales y la directiva de la Autoridad Nacional de Protección de Datos Personales sobre el tratamiento de datos
  mediante sistemas de videovigilancia.
- El análisis de **expresiones faciales** es el componente menos fiable: una expresión no equivale a una emoción y
  los modelos tienen sesgos según edad, etnia y condiciones de luz. La Ley de IA de la Unión Europea prohíbe inferir
  emociones en el trabajo y en la educación. Por eso aquí es informativo y viene **apagado** con cámaras reales
  (en el modo demo se muestra): se activa en **Ajustes**, solo con consentimiento informado.
- Las capturas y la base de datos quedan en la carpeta `datos/` de su equipo; nada se envía a internet.

## Limitaciones conocidas

- Las reglas son heurísticas sobre la pose: no aprenden de datos. Su precisión real debe medirse con el script de
  evaluación antes de afirmar cifras de desempeño.
- Abrazos, juegos bruscos o bailes cerca de otra persona pueden parecer peleas; agacharse a recoger algo puede
  parecer una caída si dura poco. Ajuste umbrales y tiempos según el lugar.
- El posible hurto depende de que el seguidor mantenga el ID del objeto; si el objeto se tapa por completo, se pierde.
- Con iluminación baja, oclusiones fuertes o personas muy lejanas (menos de unos 60 píxeles de alto) la pose
  se degrada, como advierte la literatura revisada.
- El modo demo usa personas sintéticas: prueba la lógica y el panel, no la precisión de los modelos.

## Solución de problemas

| Problema | Qué hacer |
|---|---|
| Va lento (menos de 5 fps) | En `config.yaml`, baje `tamano_imagen` a 320 y `fps_objetivo` a 6 |
| «No se pudo abrir 0» | Cierre otras apps que usen la webcam (Zoom, Teams) o pruebe con la fuente `1` |
| La cámara IP no conecta | Pruebe la URL en VLC (Medio → Abrir ubicación de red); verifique usuario, clave y que esté en la misma red |
| No aparecen emociones | Faltan los modelos ONNX: ejecute `python scripts/descargar_modelos.py` |
| Error al instalar `torch` | Use Python 3.10, 3.11 o 3.12 |

## Créditos

Modelos: YOLO11 de Ultralytics (licencia AGPL-3.0; para uso comercial requiere licencia de Ultralytics), YuNet y
el reconocedor de expresiones de OpenCV Zoo (Apache-2.0). Desarrollado con asistencia de Claude (Anthropic) como
producto de la revisión sistemática del autor.
