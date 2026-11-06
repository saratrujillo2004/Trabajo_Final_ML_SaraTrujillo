"""Configuración común de las pruebas: permite importar los módulos de mlops_pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlops_pipeline"))