"""Pruebas del monitoreo de data drift y del tablero."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import model_monitoring as monitoreo

RUTA_TABLERO = str(Path(monitoreo.__file__))


def test_psi_igual_a_cero_sin_cambios():
    serie = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert monitoreo.calcular_psi(serie, serie) == pytest.approx(0, abs=1e-6)


def test_psi_detecta_cambio_fuerte():
    rng = np.random.default_rng(0)
    referencia = pd.Series(rng.normal(0, 1, 2000))
    actual = pd.Series(rng.normal(2, 1, 2000))
    assert monitoreo.calcular_psi(referencia, actual) >= monitoreo.PSI_FUERTE


def test_jensen_shannon_entre_cero_y_uno():
    rng = np.random.default_rng(0)
    valor = monitoreo.calcular_jensen_shannon(pd.Series(rng.normal(0, 1, 500)), pd.Series(rng.normal(1, 1, 500)))
    assert 0 < valor <= 1


def test_chi_cuadrado_detecta_cambio_de_proporciones():
    referencia = pd.Series(["Empleado"] * 90 + ["Independiente"] * 10)
    actual = pd.Series(["Empleado"] * 40 + ["Independiente"] * 60)
    assert monitoreo.calcular_chi_cuadrado(referencia, actual) < 0.01


@pytest.mark.parametrize("psi, esperado", [(0.05, "Verde"), (0.15, "Amarillo"), (0.30, "Rojo")])
def test_semaforo_numerica(psi, esperado):
    assert monitoreo.semaforo_numerica(psi) == esperado


@pytest.mark.parametrize("p_valor, esperado", [(0.5, "Verde"), (0.03, "Amarillo"), (0.001, "Rojo")])
def test_semaforo_categorica(p_valor, esperado):
    assert monitoreo.semaforo_categorica(p_valor) == esperado


def tabla_alertas(rojas, amarillas):
    """Tabla de un periodo con la cantidad de alertas indicada."""
    alertas = ["Rojo"] * len(rojas) + ["Amarillo"] * len(amarillas)
    return pd.DataFrame({"variable": rojas + amarillas, "alerta": alertas})


def test_recomendaciones():
    assert "estable" in monitoreo.generar_recomendaciones(tabla_alertas([], []))[0]
    assert "Revisar" in monitoreo.generar_recomendaciones(tabla_alertas(["edad_cliente"], []))[0]
    assert "reentrenar" in monitoreo.generar_recomendaciones(tabla_alertas(["a", "b", "c"], []))[0]
    assert "reentrenar" in monitoreo.generar_recomendaciones(tabla_alertas(["prob_no_pago"], []))[0]
    assert "Vigilar" in monitoreo.generar_recomendaciones(tabla_alertas([], ["saldo_total"]))[0]


def test_calculo_de_drift_y_resumen(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoreo, "RUTA_DRIFT", tmp_path / "monitoreo_drift.csv")
    monitoreo.main()

    drift = pd.read_csv(tmp_path / "monitoreo_drift.csv")
    assert set(drift["alerta"]) <= {"Verde", "Amarillo", "Rojo"}
    assert drift.groupby("periodo")["registros"].min().min() >= monitoreo.MINIMO_REGISTROS


def test_tablero_se_ejecuta_sin_errores():
    tablero = AppTest.from_file(RUTA_TABLERO, default_timeout=120).run()
    assert not tablero.exception
    tablero.sidebar.selectbox[0].set_value(tablero.sidebar.selectbox[0].options[0]).run()
    assert not tablero.exception