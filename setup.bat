@echo off
echo === Creando entorno virtual ===
python -m venv venv

echo === Activando entorno virtual ===
call venv\Scripts\activate.bat

echo === Actualizando pip ===
python -m pip install --upgrade pip

echo === Instalando dependencias ===
pip install -r requirements.txt

echo === Entorno listo ===
pause
