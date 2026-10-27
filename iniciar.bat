@echo off
REM VigIA - inicio en Windows. Uso: doble clic, o "iniciar.bat --demo" para la escena simulada.
chcp 65001 >nul
cd /d "%~dp0"

REM Python 3.10 a 3.12: PyTorch (lo usa YOLO) no siempre tiene version para Python mas nuevo.
set PY=
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V --version >nul 2>nul && set PY=py -%%V
  )
)
if not defined PY (
  if /I "%~1"=="--demo" (
    set PY=python
  ) else (
    echo.
    echo No se encontro Python 3.10, 3.11 ni 3.12.
    echo Instale Python 3.12 con:   winget install Python.Python.3.12
    echo o desde https://www.python.org/downloads/release/python-3129/  y vuelva a ejecutar.
    goto error
  )
)
echo Usando: %PY%

if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno virtual de Python...
  %PY% -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"

if /I "%~1"=="--demo" (
  if not exist ".venv\instalado_demo.txt" if not exist ".venv\instalado.txt" (
    echo Instalando lo minimo para el modo demo...
    python -m pip install --upgrade pip
    pip install -r requirements-demo.txt
    if errorlevel 1 goto error
    echo ok > ".venv\instalado_demo.txt"
  )
) else (
  if not exist ".venv\instalado.txt" (
    echo Instalando dependencias. La primera vez tarda varios minutos...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 goto error
    echo ok > ".venv\instalado.txt"
  )
  if not exist "models\yolo11n-pose.pt" (
    echo Descargando modelos...
    python scripts\descargar_modelos.py
  )
)

echo.
echo Abriendo VigIA en http://127.0.0.1:8000  -  para detenerlo, pulse Ctrl+C o cierre esta ventana.
python -m vigia %*
goto fin

:error
echo.
echo Ocurrio un error. Revise el mensaje de arriba.
pause
:fin
