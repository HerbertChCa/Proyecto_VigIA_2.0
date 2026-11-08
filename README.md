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
| Robo de vehículo | Alguien distinto de quien estacionó la moto, bicicleta o auto se sube o se queda sobre él ≥ 3 s; alta si se lo lleva | Media / Alta |
| Robo con huida | Una mano o un pie de A llega al cuerpo de B y, en los 4 s siguientes, A corre (≥ 1.6 alturas/s) mientras B no | Alta |
| Vandalismo | Una persona sola (nadie a menos de 1.5 alturas) golpea rápido un vehículo o una zona «protegida» | Alta |
| Posible arma | Un cuchillo o un bate detectado ≥ 2 veces en 2 s a menos de 0.3 alturas de una muñeca | Alta |
| Aglomeración | 15 personas o más durante ≥ 5 s | Media |
| Huida masiva | 4 personas o más, y al menos la mitad de las presentes, corren a la vez | Alta |
| Expresión facial (opcional) | Alegría, neutral, sorpresa, tristeza, enojo, miedo o desagrado, por persona; apagada por defecto | Informativa |

Todas las distancias se miden en «alturas de persona», así que las reglas no dependen de la resolución ni de la
distancia a la cámara. Todos los umbrales se cambian en `config.yaml` o desde **Ajustes** en el panel.
Cada zona puede tener un **horario** (por ejemplo, 22:00 a 06:00, cruzando la medianoche): fuera de esa franja no alerta.

## Fundamento en el artículo de revisión

VigIA es el producto del artículo *Reconocimiento del comportamiento humano en cámaras de seguridad mediante visión
artificial y aprendizaje profundo: revisión sistemática de desafíos técnicos, percepción social y tendencias (2021-2026)*,
que analizó 48 estudios (E01-E48). Cada decisión de diseño responde a un hallazgo o a una brecha de su agenda (Tabla 11):

| Hallazgo del artículo | Decisión en VigIA | Dónde |
|---|---|---|
| PE2: dominan las CNN 2D supervisadas (47.7 %); los enfoques por esqueleto no conservan la apariencia y podrían mejorar la aceptación (E18, E39; sección 4.2) | Analizar la **pose** (17 puntos) con reglas, no la apariencia | `modelos.py`, `reglas.py` |
| El corpus estudia violencia (RWF-2000: E18, E20, E24, E30, E39), agresión y caídas con reglas en el borde (E24), merodeo con YOLO + seguimiento (E22) y robo o hurto (E17, E37) | Trece tipos de alerta con esas mismas técnicas (YOLO11 + ByteTrack + reglas sobre la pose): los siete del guion demo y seis que completan el mapa de actos maliciosos | `reglas.py` |
| PE4: el tiempo real preocupa a 36 de 44 estudios, pero solo 17 miden la velocidad; la salida es el borde y la eficiencia (E24, E34, E36, E40, E44) | Modelos *nano* en CPU, detector de objetos 1 de cada 3 frames, FPS visible en el panel y latencia medida | `modelos.py`, `camara.py` |
| Brecha metodológica: 15 de 44 reportan solo *accuracy*; falta un conjunto mínimo de métricas (AUC, F1, latencia y hardware) | La evaluación entrega esa **ficha** completa en `ficha_evaluacion.md` | `scripts/evaluar_rwf2000.py` |
| Brecha de validación: solo 2 de 44 evalúan con datos externos y el desempeño cae 4 a 10 puntos (E20, E37) | Modo `--umbral` fijo para validar en otros datos y **veredicto del operador** que mide la precisión real de cada instalación | `evaluar_rwf2000.py`, `alertas.py` |
| Brecha contextual: un solo estudio latinoamericano (E12) y solo el 9.1 % graba material propio | Funciona con cámaras RTSP reales y evalúa videos propios con la estructura `Fight/` y `NonFight/` | `camara.py`, `evaluar_rwf2000.py` |
| Brecha de explicabilidad: solo 3 estudios usan Grad-CAM (E25, E37, E41) y ninguno lo evalúa con operadores | Cada alerta trae su medida, su captura y los botones **Confirmar / Falsa alarma**, que permiten medir si la explicación ayuda | `reglas.py`, `alertas.py`, `web/` |
| Solo 6 de 44 tratan la privacidad | Rostros pixelados por defecto, procesamiento local, sin grabación continua, capturas borradas a los 7 días | `dibujo.py`, `config.yaml` |
| PE3: la aceptación es condicional al propósito (82.2 % para un fin protector, E45), el reconocimiento facial es poco aceptado (36 %, E46) y la confianza depende de la institución (E48) | Página **/aviso** con finalidad, responsable y lo que el sistema no hace; sin reconocimiento facial; emociones opcionales y apagadas por defecto | `aviso.py`, `config.yaml` |
| Brecha tecnológica: 0 de 44 usan modelos de visión-lenguaje | **Búsqueda con CLIP** multilingüe sobre las capturas; la detección sigue siendo por reglas, comparables con la ficha de evaluación | `busqueda.py` |
| PE3 y brecha contextual: aceptación condicional y un solo estudio latinoamericano | **Encuesta** de aceptación en la feria con preguntas inspiradas en E45-E48 y resultados comparados en vivo | `encuesta.py`, `web/encuesta.html` |
| Los enfoques por esqueleto no conservan la apariencia (E18, E39) | Modo **solo esqueleto**: las personas se borran de la imagen y la detección no cambia | `dibujo.py` |
| Brecha de validación: faltan pruebas con personas reales | **Retos** de la feria: cada intento suma a la tasa de detección real de cada regla | `reto.py`, `web/reto.html` |

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

   Esa dirección muestra la **entrada**: una escena animada donde personas dibujadas solo con su esqueleto
   caminan frente a una cámara y una de ellas se cae, más un resumen del producto y del artículo. El panel del
   operador está en `/panel`, los retos en `/reto` y la encuesta en `/encuesta/resultados`.

   La primera vez en cada sesión, la entrada muestra un **arranque**: se dibuja el ojo de VigIA, se revisa cada parte
   del sistema con datos reales (modelos, cámaras en línea, privacidad) y el ojo se abre sobre la escena; cualquier
   tecla lo salta. Al pasar de una página a otra, la nueva aparece barrida por una línea de escaneo y, en el panel y
   el reto, unos marcos de enfoque se fijan sobre cada cámara. Si Windows tiene activado «reducir animaciones», todo
   esto se omite.
4. **Cámara IP:** en el panel, **+ Cámara** y escriba la dirección RTSP. Ejemplos:
   - Hikvision: `rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101`
   - Dahua: `rtsp://usuario:clave@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
   - Un video: la ruta a un `.mp4` (útil para probar con clips de UCF-Crime o RWF-2000). Se analiza completo, cuadro
     a cuadro, y al terminar queda en el último cuadro (vea «Videos subidos» más abajo).
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

## En vivo con la misma precisión que en cámara lenta

Un video subido en modo **Detallado** se analiza a 10 cuadros por cada segundo de video, aunque la PC tarde más
(se ve en cámara lenta). Una cámara en vivo no puede esperar: analiza los cuadros que la PC alcanza y se salta el
resto. La precisión en vivo depende entonces de cuántos cuadros por segundo alcanza la computadora; la tarjeta
de cada cámara lo indica con «precisión alta» (7 o más), «media» (4 a 7) o «baja» (menos de 4). Al pasar el
mouse por encima se ve cuánto tarda cada parte del análisis.

Para acercar lo en vivo a lo detallado, paso a paso (mida después de cada uno):

1. **Medir:** `python scripts\medir_velocidad.py video.mp4` dice cuántos cuadros por segundo da esta PC y qué parte
   del análisis pesa más.
2. **Perfil rápido:** `python -m vigia --rapido` (o `perfil: rapido` en `config.yaml`). Personas a 384 px, objetos
   a 480 px y rostros cada 3 cuadros. De cerca detecta igual; pierde algo de alcance (objetos a más de 5-6 m).
3. **Acelerar los modelos:** `python scripts\acelerar_modelos.py` (o `--rapido` si usará ese perfil). Los exporta a
   OpenVINO, que en CPU suele ser 2 a 3 veces más rápido con las mismas detecciones. Se hace una vez, con internet;
   VigIA los usa solo (la consola dice «personas en openvino»). Si cambia los tamaños de imagen, vuelva a correrlo.
4. **Una cámara a la vez:** las cámaras se turnan los modelos; dos cámaras van a la mitad cada una.
5. **Emociones apagadas** (`analizar_emociones: false`) si no se van a mostrar: ahorran el modelo de expresiones.
6. **Con tarjeta gráfica NVIDIA**, VigIA la usa sola (instale PyTorch con CUDA): suele pasar de 30 cuadros por segundo.

En los videos subidos, el botón **Detallado / Tiempo real** de la tarjeta cambia de modo mientras corre: en tiempo
real el video va a su velocidad y se comporta como una cámara en vivo (sirve para mostrar qué se perdería en vivo
con esta PC); en detallado no se salta nada. Para la feria: `python -m vigia --feria --rapido`.

## Preparar la feria paso a paso (para que no falle en vivo)

**Segunda mirada.** Si la PC analiza pocos cuadros por segundo en vivo, VigIA guarda los últimos 15 s de la
cámara. Cuando ve algo que podría ser un incidente rápido (dos personas juntas, un movimiento brusco, alguien que
queda horizontal), vuelve a analizar ese tramo cuadro a cuadro, como un video en modo detallado, con un segundo
analizador. Lo que encuentre y la vista en vivo no haya avisado llega como alerta unos segundos después, con la
explicación «Detectado en la segunda mirada». En la tarjeta aparece «Segunda mirada: revisando… 40 %», y el botón
**Revisar 10 s** la pide a mano. En los retos, antes de decir «no te detectó», VigIA revisa los últimos segundos
del intento. Se ajusta en `config.yaml` (`procesamiento.revision`).

**El día anterior**
1. `python scripts\acelerar_modelos.py --rapido` (con internet) y `python scripts\medir_velocidad.py video.mp4 --rapido`.
2. Grabe con el celular, en un lugar parecido, cada reto que mostrarán y páselo por
   `python scripts\diagnosticar_video.py`: si alguno no da alerta, ahí se ve por qué.

**Al llegar a la sala** (`python -m vigia --feria --rapido`, una sola cámara)
1. Fije la cámara (trípode o apoyo), de lado a la zona de pruebas, sin ventanas detrás de las personas, a una
   distancia en que se vea el cuerpo entero.
2. Dibuje las zonas (botón **Zonas**): una restringida y una de merodeo, sobre el piso.
3. Abra **Chequeo** y corrija lo que salga en rojo. Párense donde harán las pruebas, caminen, muestren la mochila
   y simulen una caída hasta que todo quede en verde o amarillo.
4. Haga cada reto una vez antes de abrir al público. Después pulse **Borrar historial**.

**Durante la feria:** mire de vez en cuando la etiqueta de precisión de la tarjeta. Si alguien hizo el reto y no
salió, **Revisar 10 s** lo vuelve a analizar cuadro a cuadro.

## Analizar un video sin tocar el código

En el panel, **+ Cámara** tiene dos pestañas:

- **Video de mi PC:** elija o arrastre un archivo (MP4, AVI, MOV, MKV o WEBM, hasta 2 GB). Se copia a
  `datos/videos/`, se analiza y se borra al quitar la cámara.

**Videos subidos.** Un video se analiza a 10 cuadros por cada segundo de video (`fps_objetivo`), aunque la
computadora sea lenta: si no le da para más, el video avanza en cámara lenta (la tarjeta dice, por ejemplo,
«cámara lenta ×0.30») en vez de saltarse cuadros. Así un tirón de mochila de 2 s se ve en 20 cuadros y no en 2 o 3.
Con el botón ⏸ de la tarjeta se pausa; al terminar, el video queda en el último cuadro y ↻ lo vuelve a empezar
(desde cero: personas, objetos y reglas, sin borrar las alertas). Para que se repita solo, por ejemplo en una feria,
ponga `repetir_videos: true` en `config.yaml`. Con varios videos a la vez cada uno va más lento, porque comparten la
computadora: para una prueba, deje solo el que quiere ver.
- **Cámara o enlace:** el número de la webcam, una dirección RTSP o el enlace **directo** a un video (`…/video.mp4`)
  o a una transmisión (`…/stream.m3u8`).
- **Enlaces de YouTube, Facebook o TikTok:** son páginas, no videos, así que VigIA los descarga con yt-dlp (hasta
  20 min, 720p, sin audio) en `datos/videos/` y los agrega como cámara; el archivo se borra al quitarla. Instale una
  vez:

  ```bat
  .venv\Scripts\pip install -r requirements-enlaces.txt
  winget install DenoLand.Deno
  ```

  Deno es el intérprete de JavaScript que yt-dlp necesita para YouTube. Si un día deja de descargar, actualice
  con `.venv\Scripts\pip install -U "yt-dlp[default]"`, porque YouTube cambia a menudo. Las condiciones de esas
  plataformas no permiten descargar sin permiso del autor: úselo solo para pruebas internas y no suba los videos
  al repositorio.

## Modo feria: retos, encuesta y solo esqueleto

```bat
iniciar.bat --feria
```

- **Reto** (`/reto`, se abre solo): el visitante elige un reto (simular una caída, entrar a la zona prohibida,
  merodear, abandonar una mochila, llevarse la de otro o sonreír), hay una cuenta regresiva y VigIA intenta
  detectarlo. Muestra en cuántos segundos lo detectó y **por qué** (la explicación de la alerta). El marcador
  acumula la tasa de detección real de la feria. Los tiempos se acortan (merodeo 10 s, abandono 8 s, en el
  suelo 3 s). Dibuje antes en **Zonas** una zona restringida y una de merodeo sobre el piso del stand y ponga una
  colchoneta para el reto de la caída.
- **Encuesta** (`/encuesta`): anónima, solo para mayores de edad, con las preguntas de aceptación de la PE3 del
  artículo. Los visitantes escanean el QR con su celular; `/encuesta/resultados` muestra los resultados en vivo,
  comparados con E45 (82.2 %) y E46 (36 %). **Exportar CSV** descarga las respuestas para el informe.
- **Solo esqueleto** (botón del panel y del reto): borra a las personas de la imagen y deja solo su esqueleto,
  también en las capturas. La detección funciona igual porque las reglas usan la pose.

Con `--feria` el servidor escucha en la red local, pero desde otros equipos **solo** se abren la encuesta, sus
resultados y el aviso; el panel, los ajustes y las alertas siguen siendo solo de la PC de VigIA. Los celulares deben
estar en la misma red Wi-Fi que la PC. Muchas redes de universidades aíslan a los dispositivos entre sí: si el QR
no abre, use el punto de acceso de un celular o un router propio. La primera vez, Windows pregunta si permite el
acceso de Python a la red: permítalo en redes privadas.

## Búsqueda con CLIP (opcional)

```bat
.venv\Scripts\pip install -r requirements-busqueda.txt
```

Active **Ajustes → Búsqueda con CLIP**. La primera vez descarga el modelo multilingüe
`xlm-roberta-base-ViT-B-32` (~1.5 GB, necesita internet). Luego, **Buscar** encuentra capturas de alertas
escribiendo lo que se ve, en español o en inglés: «persona con mochila», «moto», «alguien en el suelo». Es la
línea futura del artículo (0 de 44 estudios usan modelos de visión-lenguaje). Solo indexa capturas de alertas, no
video continuo; las capturas tomadas en modo solo esqueleto no muestran a las personas, así que la búsqueda solo
ve la escena.

## Menos falsas alarmas

- **Escenario aprendido.** Solo puede quedar «abandonado» un objeto que alguien trajo: lo que se ve en los primeros
  10 s, lo que aparece sin nadie al lado (prendas de un escaparate, un estante) o lo que ya estaba en ese lugar se
  considera parte del escenario.
- **El operador enseña.** Marcar como **Falsa alarma** una alerta de objeto abandonado, hurto o robo de vehículo hace
  que ese objeto, en ese lugar de la imagen, no vuelva a alertar. Se guarda en `datos/ignorar.json`.
- **Alertas agrupadas.** Si la misma situación (cámara, tipo y severidad) se repite en menos de 20 s mientras la
  alerta sigue pendiente, se suma a ella con un contador «×N» y la tarjeta vuelve al inicio de la lista
  (`alertas.agrupar`). En modo feria no se agrupan: cada intento del reto es su propia alerta.
- **Borrar historial.** El botón del panel borra las alertas y sus fotos y reinicia lo aprendido de la escena
  (escenario y falsas alarmas). Úselo entre una prueba y otra; las cámaras, zonas y ajustes se conservan.
- **Merodeo sin zonas, opcional.** Medir el merodeo en toda la cámara cuando no hay zonas está apagado por defecto,
  porque en una tienda o un pasillo cualquiera se queda más de un minuto. Se activa en Ajustes.

## Evaluación con videos propios: todas las reglas

`scripts/evaluar_videos.py` pasa videos etiquetados por el mismo flujo que una cámara y entrega, por regla,
sensibilidad, precisión, F1 y especificidad con su intervalo de confianza del 95 %, el tiempo hasta la detección y
las falsas alarmas por hora en escenas cotidianas. Organice los videos con una carpeta por tipo de alerta:

```
evaluacion/
  caida/positivos/        caidas reales o actuadas
  caida/normales/         parecidos sin caída: agacharse, sentarse
  robo_vehiculo/positivos/
  robo_vehiculo/normales/ el dueño que se sube a su moto
  normales/               escenas cotidianas sin incidentes
```

```bat
python scripts/evaluar_videos.py --datos evaluacion --operador datos/alertas.db
```

Escribe `ficha_videos.md`, lista para el informe, y `resultados_videos.csv` con el detalle de cada video. Con
`--operador` agrega la precisión observada con los veredictos del panel (uso real). Con pocos videos por regla los
intervalos son amplios: reporte siempre la cifra con su intervalo y el número de videos.

## Diagnosticar una prueba que falló

Si una prueba no dio la alerta esperada, pase el video grabado por `scripts/diagnosticar_video.py`. Usa el mismo
flujo que una cámara y explica qué vio VigIA: si detectó la mochila y con qué confianza, quién la llevaba en cada
momento, si hubo cuerpo a cuerpo, la velocidad de las extremidades y cuánto se movió la cámara.

```bat
python scripts\diagnosticar_video.py VID_20261105_155016.mp4
```

Deja en `diagnostico/` el video anotado y un CSV con una fila por cuadro. **Grabe las pruebas con el celular
quieto** (apoyado o en un trípode): con la cámara en la mano las zonas dejan de coincidir con el piso y las
velocidades de las personas en la imagen no son las reales.

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
precisión con los veredictos, que se detecten el robo con huida, el vandalismo, las armas en la mano, la
aglomeración y la huida masiva (y no sus casos parecidos: irse caminando, golpear con otra persona cerca, un
cuchillo sobre una mesa), que las zonas respeten su horario, que el AUC del script de evaluación sea correcto y que el aviso declare la finalidad.

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
- El posible hurto depende de que el detector vea el objeto: una mochila de espaldas, tapada por el cuerpo o
  movida muy rápido puede no detectarse. El arrebato (quitarle a alguien lo que lleva encima) se sigue aunque el
  objeto cambie de número, pero no si deja de verse por completo antes del tirón.
- La ropa solo se reconoce con el detector de vocabulario abierto (`python scripts/preparar_vocabulario.py`); el de
  COCO no tiene esa clase. Una prenda se vigila si alguien la trae en la mano y en movimiento; la de un perchero o
  un escaparate es parte del lugar y no alerta.
- El forcejeo se reconoce por los cuerpos juntos con empujones; un abrazo quieto no alerta, pero un juego brusco
  o un baile pegado sí pueden hacerlo. Cada alerta debe revisarse en la captura.
- Con iluminación baja, oclusiones fuertes o personas muy lejanas (menos de unos 60 píxeles de alto) la pose
  se degrada, como advierte la literatura revisada.
- La detección de armas usa las clases «knife» y «baseball bat» del modelo COCO: a distancia o con poca luz confunde
  objetos alargados (paraguas, celulares, herramientas) y no detecta armas de fuego. Cada alerta debe verificarse
  en la captura; es una señal para el operador, no una prueba.
- Robo con huida y huida masiva dependen de que el seguidor mantenga el ID de quien corre; un corredor que sale
  del cuadro en menos de un segundo puede no alcanzar a medirse.
- El modo demo usa personas sintéticas: prueba la lógica y el panel, no la precisión de los modelos.

## Solución de problemas

| Problema | Qué hacer |
|---|---|
| Va lento (menos de 5 fps) o dice «precisión baja» | Siga «En vivo con la misma precisión que en cámara lenta»: `scripts\medir_velocidad.py`, `--rapido` y `scripts\acelerar_modelos.py` |
| Un video subido no da alertas | Mire la tarjeta: si dice «cámara lenta», el análisis va bien, solo más despacio; espere a que pase la escena. Analice un video a la vez y quite las demás cámaras. Para ver qué detectó cuadro por cuadro: `python scripts\diagnosticar_video.py video.mp4` |
| «No se pudo abrir 0» | Cierre otras apps que usen la webcam (Zoom, Teams) o pruebe con la fuente `1` |
| La cámara IP no conecta | Pruebe la URL en VLC (Medio → Abrir ubicación de red); verifique usuario, clave y que esté en la misma red |
| No aparecen emociones | Faltan los modelos ONNX: ejecute `python scripts/descargar_modelos.py` |
| Error al instalar `torch` | Use Python 3.10, 3.11 o 3.12 |
| La mochila o la prenda aparece y desaparece | Encuadre a la persona entera a 3-5 m, con buena luz y la cámara quieta; si la PC aguanta, ponga `objetos_cada_n_frames: 2`. Revise con `scripts/diagnosticar_video.py` qué confianza le da el detector |

## Créditos

Modelos: YOLO11 de Ultralytics (licencia AGPL-3.0; para uso comercial requiere licencia de Ultralytics), YuNet y
el reconocedor de expresiones de OpenCV Zoo (Apache-2.0). Desarrollado con asistencia de Claude (Anthropic) como
producto de la revisión sistemática del autor.
