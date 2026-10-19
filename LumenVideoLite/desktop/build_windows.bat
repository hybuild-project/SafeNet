@echo off
setlocal
cd /d "%~dp0"

py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --clean --onefile --windowed --name LumenVideoLite --paths .. --paths ..\.. --add-data "..\video_search_lite.html;." --add-data "..\..\models\320n.onnx;models" --add-data "..\..\models\640m.onnx;models" --add-data "..\..\models\top_yolov8s.onnx;models" --collect-all cv2 --collect-all onnxruntime launcher.py
if errorlevel 1 exit /b 1

echo EXE creato in desktop\dist\LumenVideoLite.exe
