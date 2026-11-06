"""Pruebas de la ingeniería de características."""

import numpy as np
import pandas as pd

from ft_engineering import (
    VARIABLE_OBJETIVO,
    aplicar_reglas_validacion,
    construir_preprocesador,
    crear_atributos_derivados,
    preparar_datos,
)


def cliente_base(**cambios):
    """Crea una tabla con un cliente válido, modificando los campos indicados."""
    datos = {
        "tipo_credito": 4, "capital_prestado": 2000000.0, "plazo_meses": 12, "edad_cliente": 35.0,
        "tipo_laboral": "Empleado", "salario_cliente": 3000000.0, "total_otros_prestamos": 600000.0,
        "cuota_pactada": 300000.0, "puntaje_datacredito": 780.0, "cant_creditosvigentes": 2,
        "huella_consulta": 4, "saldo_mora": 0.0, "saldo_total": 1500000.0,
        "creditos_sectorFinanciero": 1, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
        "promedio_ingresos_datacredito": 1200000.0, "tendencia_ingresos": "Creciente",
    }
    datos.update(cambios)
    return pd.DataFrame([datos])


def test_reglas_convierten_valores_invalidos_en_nulos():
    df = cliente_base(edad_cliente=123, puntaje_datacredito=999, salario_cliente=0,
                    tipo_laboral="Pensionado", tendencia_ingresos="8315")
    resultado = aplicar_reglas_validacion(df)
    columnas = ["edad_cliente", "puntaje_datacredito", "salario_cliente", "tipo_laboral", "tendencia_ingresos"]
    assert resultado[columnas].isna().all(axis=None)


def test_reglas_conservan_valores_validos():
    resultado = aplicar_reglas_validacion(cliente_base())
    assert resultado.notna().all(axis=None)


def test_reglas_filtran_variable_objetivo_invalida():
    df = pd.concat([cliente_base(), cliente_base()], ignore_index=True)
    df[VARIABLE_OBJETIVO] = [1, 5]
    resultado = aplicar_reglas_validacion(df)
    assert len(resultado) == 1


def test_reglas_funcionan_sin_variable_objetivo():
    resultado = aplicar_reglas_validacion(cliente_base())
    assert len(resultado) == 1


def test_atributos_derivados():
    resultado = crear_atributos_derivados(cliente_base(saldo_mora=5000, tendencia_ingresos=np.nan, tipo_credito=68))
    fila = resultado.iloc[0]
    assert fila["ratio_cuota_salario"] == 0.1
    assert fila["ratio_endeudamiento"] == 0.2
    assert fila["tiene_mora"] == 1
    assert fila["tendencia_sin_info"] == 1
    assert fila["tipo_credito"] == "Otros"


def test_tipo_credito_frecuente_se_conserva():
    resultado = crear_atributos_derivados(cliente_base(tipo_credito=9))
    assert resultado.iloc[0]["tipo_credito"] == "9"


def test_preparar_datos_division_estratificada():
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    assert len(X_train) + len(X_test) == 10763
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert "puntaje" not in X_train.columns


def test_preprocesador_transforma_sin_nulos():
    X_train, _, _, _, _ = preparar_datos()
    transformado = construir_preprocesador().fit_transform(X_train)
    assert transformado.shape[0] == len(X_train)
    assert not np.isnan(transformado).any()