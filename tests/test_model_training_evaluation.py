"""Pruebas del entrenamiento y la evaluación de modelos."""

import numpy as np
import pandas as pd
import pytest

import model_training_evaluation as entrenamiento


def test_summarize_classification_clase_no_pago():
    y_real = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.9, 0.4, 0.2, 0.1])
    metricas = entrenamiento.summarize_classification(y_real, y_pred, y_proba)
    assert metricas["accuracy"] == pytest.approx(0.75)
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(0.5)
    assert metricas["auc"] == pytest.approx(1.0)


def test_definir_modelos_con_ponderacion():
    modelos = entrenamiento.definir_modelos(pd.Series([0] * 5 + [1] * 95))
    assert set(modelos) == {"Regresión logística", "Árbol de decisión", "Random forest", "XGBoost"}
    assert modelos["XGBoost"].get_params()["scale_pos_weight"] == pytest.approx(5 / 95)


def test_flujo_completo_guarda_resultados(tmp_path, monkeypatch):
    monkeypatch.setattr(entrenamiento, "CARPETA_RESULTADOS", tmp_path)
    monkeypatch.setattr(entrenamiento, "RUTA_MODELO", tmp_path / "mejor_modelo.joblib")
    entrenamiento.main()

    tabla = pd.read_csv(tmp_path / "tabla_resumen.csv")
    assert len(tabla) == 4
    assert tabla["auc"].between(0.5, 1).all()
    for archivo in ["mejor_modelo.joblib", "comparacion_metricas.png", "curvas_roc.png"]:
        assert (tmp_path / archivo).exists()